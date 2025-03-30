import os
import json
import heapq
import torch
import torch.nn as nn
from torch.distributions import Categorical
import concurrent.futures

# Import the CliffordTableau_torch class
from tableau import CliffordTableau
# Import the forward MLP models
from models import DiscreteUniform, MLP

##############################################################################
#  Forward Mask
##############################################################################
def forward_mask(logits, action_mapping, last_gate_mapping):
    """
    Apply a forward mask to the logits with the following rules:

    1. Disallow two-qubit gates (e.g., 'CNOT', 'SWAP') if both qubits are 'untouched'
       (i.e., if neither qubit is in last_gate_mapping).
    2. Always allow the 'terminal' action (never mask it out).
    3. Disallow repeating the same two-qubit gate on the exact same pair of qubits.
       (i.e., if the last gate for that pair was exactly the same gate).
    4. Disallow any single-qubit gate immediately after a single-qubit gate on the
       same qubit. (i.e., if the last gate for that qubit was also single-qubit,
       then mask out another single-qubit gate on that qubit.)
    """

    # For convenience, define sets of gate names
    single_qubit_gates = {"I","H", "S", "HS", "SH", "HSH"}  # or however many you have
    two_qubit_gates    = {"CNOT", "SWAP"}

    # We'll modify logits in-place; make sure we do not destroy the original if needed
    for idx, action in action_mapping.items():
        gate_name = action[0]

        # ------------------------------------------------------
        # (2) Always allow 'terminal' -> do NOT mask it out
        # ------------------------------------------------------
        if gate_name == "terminal":
            continue  # skip any masking

        # ------------------------------------------------------
        # Identify the qubits involved in this action
        # ------------------------------------------------------
        if gate_name in single_qubit_gates:
            # Example: ("H", q)
            q = action[1]
            qubits_involved = [q]

        elif gate_name in two_qubit_gates:
            # Example: ("CNOT", q1, q2)
            q1, q2 = action[1], action[2]
            qubits_involved = [q1, q2]

        else:
            # If it's something else unexpected, you can decide how to handle
            # or simply continue
            continue

        # ------------------------------------------------------
        # (1) Disallow two-qubit gates if both qubits are untouched
        # ------------------------------------------------------
        if gate_name in two_qubit_gates:
            q1, q2 = qubits_involved
            if (q1 not in last_gate_mapping) and (q2 not in last_gate_mapping):
                logits[idx] = float('-inf')
                continue

        # ------------------------------------------------------
        # (3) Disallow repeating the same two-qubit gate on exact same qubits
        #     If you track pairs under something like last_gate_mapping[(q1,q2)],
        #     be sure you keep (q1,q2) sorted or consistent so (1,0) matches (0,1)
        # ------------------------------------------------------
        if gate_name in two_qubit_gates:
            # For consistency, sort qubits if you consider (0,1) == (1,0)
            # so that we store/look up the same key.
            q1, q2 = sorted(qubits_involved)
            pair_key = (q1, q2)

            # If the *exact same* gate was last used on this pair, disallow
            last_pair_gate = last_gate_mapping.get(pair_key, None)
            if last_pair_gate == gate_name:
                logits[idx] = float('-inf')
                continue

        # ------------------------------------------------------
        # (4) Disallow single-qubit gate if the last gate for that qubit
        #     was also a single-qubit gate.
        # ------------------------------------------------------
        if gate_name in single_qubit_gates:
            q = qubits_involved[0]
            last_qubit_gate = last_gate_mapping.get(q, None)

            # Check if the last gate on that qubit is also single-qubit
            if last_qubit_gate in single_qubit_gates:
                logits[idx] = float('-inf')
                continue

    return logits


##############################################################################
#  Backward Mask
##############################################################################
def backward_mask(logits, action_mapping, gate_history, disallow_terminal=True):
    """
    Mask out invalid backward actions that can be 'reversed'
    based on the gate_history. (Optional logic for a "reverse" pass.)
    """
    masked_logits = logits.clone().detach()
    masked_logits[:] = float('-inf')

    last_gate_index_per_qubit = {}
    for i, gate in enumerate(gate_history):
        gate_qubits = gate[1:]
        for qubit in gate_qubits:
            last_gate_index_per_qubit[qubit] = i

    # A gate is "most recent" if it's the last one applied to all its qubits
    for i, gate in enumerate(gate_history):
        gate_qubits = gate[1:]
        is_most_recent = True
        for q in gate_qubits:
            if last_gate_index_per_qubit[q] != i:
                is_most_recent = False
                break
        if is_most_recent:
            # unmask the action that corresponds to 'gate'
            for idx, a in action_mapping.items():
                if a == gate:
                    masked_logits[idx] = logits[idx]
                    break

    # allow terminal if disallow_terminal=False
    if not disallow_terminal:
        for idx, a in action_mapping.items():
            if a[0] == "terminal":
                masked_logits[idx] = logits[idx]
                break

    return masked_logits


##############################################################################
#  Action Mapping (includes I,H,S,HS,SH,HSH for each qubit)
##############################################################################
def action_mapping(n_qubits):
    """
    index -> action tuple. Now includes:
      - ("terminal",)
      - ("H", q), ("S", q), ("HS", q), ("SH", q), ("HSH", q) for q in [0..n_qubits-1]
      - two-qubit gates: "CNOT", "SWAP" for pairs
    """
    mapping = {}
    idx = 0

    # 1) Terminal
    mapping[idx] = ("terminal",)
    idx += 1

    # 2) Single-qubit gates (6 types) for each qubit
    single_qubit_gates = ["H","S","HS","SH","HSH"]
    for q in range(n_qubits):
        for gname in single_qubit_gates:
            mapping[idx] = (gname, q)
            idx += 1

    # 3) Two-qubit gates. Example:
    #    We'll do fully connected or just pairs? Below we do a simplified approach:
    #    for each pair (i < j), we add:
    #      ("CNOT", i, j), ("CNOT", j, i), ("SWAP", i, j)
    #    Adjust as needed.
    for i in range(n_qubits-1):
            mapping[idx] = ("CNOT", i, i+1)
            idx += 1
            #mapping[idx] = ("CNOT", j, i)
            #idx += 1
            mapping[idx] = ("SWAP", i, i+1)
            idx += 1
        #for j in range(i+1, n_qubits):
        #    mapping[idx] = ("CNOT", i, j)
        #    idx += 1
            #mapping[idx] = ("CNOT", j, i)
            #idx += 1
        #    mapping[idx] = ("SWAP", i, j)
        #    idx += 1

    return mapping


##############################################################################
#  Cost Functions
##############################################################################
def cal_cost(trajectories, pauli_str_list=["II", "XX", "YY", "ZZ"], w_list=[0.25, 0.25, 0.25, 0.25], epsilon=1.0):
    """
    COST_epsilon({U_i}) = sum_{P} [ w_P * product_{i} exp(-epsilon^2 * p_i(P)/2 ) ]

    We implement a simplified version by summing each p_i(P) over i, then do exp(- eps^2/2 * sum_i p_i(P)).
    """
    if not trajectories:
        return torch.tensor(0.0)

    device = trajectories[0].device
    num_traj = len(trajectories)
    num_paulis = len(pauli_str_list)

    epsilon = torch.tensor(epsilon, dtype=torch.float64, device=device)
    p_values = torch.empty((num_traj, num_paulis), dtype=torch.float64, device=device)

    # Fill p_values with prob_P
    for j, p_str in enumerate(pauli_str_list):
        for i, traj in enumerate(trajectories):
            val = traj.prob_P(p_str)  # now returns a Python float
            p_values[i, j] = val

    sum_p_values = p_values.sum(dim=0)  # shape (num_paulis,)
    exponents = - (epsilon.pow(2) * sum_p_values / 2.0)
    product_terms = torch.exp(exponents)  # shape (num_paulis,)

    w_tensor = torch.tensor(w_list, dtype=torch.float64, device=device).abs()
    total_cost = torch.dot(w_tensor, product_terms)
    return total_cost


def cal_empirical_average(trajectories, pauli_str):
    """
    Empirical average over trajectories:
      \hat{o}(P) = sum_i [ p_i(P) * empirical_value_i(P) ] / sum_i [ p_i(P) ].

    p_i(P) = prob_P, empirical_value_i(P) = empirical_sample(P).
    """
    if not trajectories:
        return 0.0
    device = trajectories[0].device
    n_traj = len(trajectories)

    prob_values = torch.empty(n_traj, dtype=torch.float64, device=device)
    emp_values  = torch.empty(n_traj, dtype=torch.float64, device=device)

    for i, traj in enumerate(trajectories):
        prob_val = traj.prob_P(pauli_str)         # float
        sample_val = traj.empirical_sample(pauli_str)  # float
        prob_values[i] = prob_val
        emp_values[i]  = sample_val

    denom = prob_values.sum()
    if denom == 0.0:
        return 0.0
    num = (prob_values * emp_values).sum()
    return (num / denom).item()  # final as Python float


def emripical_average(trajectories, pauli_str_list=["II","XX","YY","ZZ"], w_list=[0.25,0.25,0.25,0.25]):
    """
    Weighted sum of empirical averages:
      sum_{p} [ w_p * cal_empirical_average(p) ].
    """
    total = 0.0
    for p_str, w in zip(pauli_str_list, w_list):
        total += w * cal_empirical_average(trajectories, p_str)
    return total


def offset_cost(trajectories, w_tensor, epsilon):
    """
    offset = sum_P w_P * exp(-N * epsilon^2/2), 
    for N= #trajectories. Typically used to shift/scale cost.
    """
    if not trajectories:
        return torch.tensor(0.0), torch.tensor(0.0)

    device = trajectories[0].device
    N = len(trajectories)
    w_tensor = w_tensor.abs()
    w_sum = w_tensor.sum()

    eps_sq = epsilon.pow(2)
    factor = torch.exp(- (N * eps_sq) / 2.0).to(device)
    offset_val = w_sum * factor
    return offset_val, w_sum


def loss_fn(pf_model, samples, forward_flow, backward_flow, off_val = 0.0, wsum = 1.0,
            pauli_str_list=["II","XX","YY","ZZ"], w_list=[0.25,0.25,0.25,0.25],
            beta=1.0, epsilon=1.0, p=10.0):
    """
    A sample loss function that includes the difference between cost and offset, plus flows.

    Args:
        pf_model: forward model with an attribute pf_model.logZ (some normalizing constant).
        samples: list of CliffordTableau_torch objects.
        forward_flow, backward_flow: lists of log-probs (tensors) from sample_clifford_trajectories.
        ...
    Returns:
        A scalar loss (torch.Tensor).
    """
    device = forward_flow[0].device
    w_tensor = torch.tensor(w_list, dtype=torch.float64, device=device)

    # offset, w_sum
    cost_val = cal_cost(samples, pauli_str_list, w_list, epsilon=epsilon).to(device)

    x = (cost_val - off_val) / (wsum - off_val + 1e-12)
    # e.g. reward = beta * exp(-p*x)
    reward = beta * torch.exp(-p * x)

    logZ = pf_model.logZ  # some learnable param
    total_P_F = torch.sum(torch.stack(forward_flow))  # sum of forward log-probs
    total_P_B = torch.sum(torch.stack(backward_flow)) # sum of backward log-probs

    # final loss
    loss = (logZ + total_P_F - reward - total_P_B).pow(2)
    print(f"reward: {reward.item():.6f}, cost: {cost_val.item():.6f}, offset: {off_val.item():.6f}")
    return loss


##############################################################################
#  Utility to Print Empirical Averages for 2-qubit or 4-qubit
##############################################################################
def print_reward_2qubit(batch_samples, i, pauli_str_list=["II","XX","YY","ZZ"], w_list=[0.25,0.25,0.25,0.25]):
    # Compute individual empirical averages for XX, YY, ZZ
    empirical_averages = {
        op: cal_empirical_average(batch_samples, op)
        for op in ["XX","YY","ZZ"]
    }
    average = emripical_average(batch_samples, pauli_str_list, w_list)
    print(f"Batch {i+1}, Empirical Averages for XX, YY, ZZ => "
          f"{empirical_averages['XX']:.4f}, "
          f"{empirical_averages['YY']:.4f}, "
          f"{empirical_averages['ZZ']:.4f},  WeightedAvg={average:.4f}")


def print_reward_4qubit(batch_samples, i, pauli_str_list=None, w_list=None):
    # Example of for 4 qubits: "XXYY", "IZIZ", "ZZII" etc. 
    # Adjust as needed for your use case
    if pauli_str_list is None:
        pauli_str_list = ["XXYY","IZIZ","ZZII"]
    if w_list is None:
        w_list = [1./3., 1./3., 1./3.]
    empirical_averages = {
        op: cal_empirical_average(batch_samples, op)
        for op in pauli_str_list
    }
    average = emripical_average(batch_samples, pauli_str_list, w_list)
    # Print them
    print(f"Batch {i+1}, Empirical Averages => ", end='')
    for op in pauli_str_list:
        print(f"{op}: {empirical_averages[op]:.4f}, ", end='')
    print(f" WeightedAvg={average:.4f}")



##############################################################################
#  Utility: Counting Actions in the extended mapping
##############################################################################
def count_actions(n_qubits):
    """
    Just returns the size of the extended action space = len(action_mapping(n_qubits)).
    """
    return len(action_mapping(n_qubits))


# -----------------------------------------------------------------------------
# 1) Helper to re-run a single trajectory of action indices with current models
# -----------------------------------------------------------------------------
def run_actions_with_current_model(
    action_seq,  # list of action indices for this trajectory
    n_qubits,
    pf_model,
    pb_model,
    action_mapping,
    device
):
    """
    Re-run a stored sequence of action indices on a new CliffordTableau,
    using the *current* pf_model/pb_model to compute flows. Return the new
    (tableau, forward_flow, backward_flow).
    """
    tableau = CliffordTableau(n_qubits, device="cpu")
    forward_flow = torch.tensor(0.0, dtype=torch.float32, device=device)
    backward_flow = torch.tensor(0.0, dtype=torch.float32, device=device)

    last_gate_mapping = {}
    for idx in action_seq:
        flat_t = tableau.to_flat_tensor().to(device)
        #W_matrix = tableau.get_heisenberg_matrix().to(device)
        #phase_vec = tableau.get_heisenberg_phase_vec().to(device)

        # forward pass
        logits_f = pf_model(flat_t) #(W_matrix, phase_vec)
        logits_f = forward_mask(logits_f.clone(), action_mapping, last_gate_mapping)
        dist_f = Categorical(logits=logits_f)
        
        # Because we're "replaying," we do NOT sample => we *force* the stored action
        chosen_action_idx = torch.tensor(idx, device=device)
        forward_flow += dist_f.log_prob(chosen_action_idx)

        action = action_mapping[idx]
        # if not terminal => do backward pass
        if action[0] != "terminal":
            logits_b = pb_model(flat_t) #(phase_vec)
            logits_b = forward_mask(logits_b.clone(), action_mapping, last_gate_mapping)
            # disallow terminal
            for i_b, a_b in action_mapping.items():
                if a_b[0] == "terminal":
                    logits_b[i_b] = float('-inf')
            dist_b = Categorical(logits=logits_b)
            backward_flow += dist_b.log_prob(chosen_action_idx)

        # apply gate
        if action[0] == "terminal":
            break
        else:
            gate_name = action[0]
            if gate_name in ["H","S","HS","SH","HSH"]:
                q = action[1]
                if gate_name == "H":
                    tableau.apply_H(q)
                elif gate_name == "S":
                    tableau.apply_S(q)
                elif gate_name == "HS":
                    tableau.apply_HS(q)
                elif gate_name == "SH":
                    tableau.apply_SH(q)
                elif gate_name == "HSH":
                    tableau.apply_HSH(q)
                last_gate_mapping[q] = gate_name

            elif gate_name == "CNOT":
                ctrl, tgt = action[1], action[2]
                tableau.apply_CNOT(ctrl, tgt)
                last_gate_mapping[ctrl] = "CNOT"
                last_gate_mapping[tgt] = "CNOT"

            elif gate_name == "SWAP":
                q1, q2 = action[1], action[2]
                tableau.apply_SWAP(q1, q2)
                last_gate_mapping[q1] = "SWAP"
                last_gate_mapping[q2] = "SWAP"

    return tableau, forward_flow, backward_flow


# -----------------------------------------------------------------------------
# 2) Modify the sampler to store action sequences
# -----------------------------------------------------------------------------
def run_trajectory(n_qubits, max_layer, pf_model, pb_model, action_mapping, device):
    tableau = CliffordTableau(n_qubits, device="cpu")
    forward_flow = torch.tensor(0.0, dtype=torch.float32, device=device)
    backward_flow = torch.tensor(0.0, dtype=torch.float32, device=device)
    last_gate_mapping = {}
    action_seq = []

    for layer in range(max_layer):
        flat_t = tableau.to_flat_tensor().to(device)

        # forward pass
        logits_f = pf_model(flat_t)
        logits_f = forward_mask(logits_f.clone(), action_mapping, last_gate_mapping)
        dist_f = Categorical(logits=logits_f)
        action_idx = dist_f.sample()
        forward_flow += dist_f.log_prob(action_idx)
        action_seq.append(action_idx.item())

        action = action_mapping[action_idx.item()]

        # backward pass if not terminal
        if action[0] != "terminal":
            logits_b = pb_model(flat_t)
            logits_b = forward_mask(logits_b.clone(), action_mapping, last_gate_mapping)
            for i_b, a_b in action_mapping.items():
                if a_b[0] == "terminal":
                    logits_b[i_b] = float('-inf')
            dist_b = Categorical(logits=logits_b)
            backward_flow += dist_b.log_prob(action_idx)

        if action[0] == "terminal":
            break

        # apply gate
        gate_name = action[0]
        if gate_name in ["H", "S", "HS", "SH", "HSH"]:
            q = action[1]
            if gate_name == "H":
                tableau.apply_H(q)
            elif gate_name == "S":
                tableau.apply_S(q)
            elif gate_name == "HS":
                tableau.apply_HS(q)
            elif gate_name == "SH":
                tableau.apply_SH(q)
            elif gate_name == "HSH":
                tableau.apply_HSH(q)
            last_gate_mapping[q] = gate_name

        elif gate_name == "CNOT":
            ctrl, tgt = action[1], action[2]
            tableau.apply_CNOT(ctrl, tgt)
            last_gate_mapping[ctrl] = "CNOT"
            last_gate_mapping[tgt] = "CNOT"

        elif gate_name == "SWAP":
            q1, q2 = action[1], action[2]
            tableau.apply_SWAP(q1, q2)
            last_gate_mapping[q1] = "SWAP"
            last_gate_mapping[q2] = "SWAP"

    return tableau, forward_flow, backward_flow, action_seq

def sample_clifford_trajectories(
    n_measurement, max_layer, n_qubits, pf_model, pb_model, action_mapping, device
):
    """
    Return: 
      - samples        (list of final tableaus)
      - forward_flow   (list of total forward log-probs)
      - backward_flow  (list of total backward log-probs)
      - action_seqs    (list of lists of indices chosen)
    """
    futures = []
    samples = []
    forward_flow_list = []
    backward_flow_list = []
    action_seqs = []

    # Using ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling issues
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _ in range(n_measurement):
            futures.append(
                executor.submit(
                    run_trajectory,
                    n_qubits,
                    max_layer,
                    pf_model,
                    pb_model,
                    action_mapping,
                    device
                )
            )

        for fut in concurrent.futures.as_completed(futures):
            tab, ff, bf, acts = fut.result()
            samples.append(tab)
            forward_flow_list.append(ff)
            backward_flow_list.append(bf)
            action_seqs.append(acts)

    return samples, forward_flow_list, backward_flow_list, action_seqs


# -----------------------------------------------------------------------------
# 3) Replay function that re-runs an entire batch with current model
# -----------------------------------------------------------------------------
def replay_batch_with_current_model(
    batch_action_seqs,  # list of action-seqs (one per trajectory)
    n_qubits,
    pf_model,
    pb_model,
    action_mapping,
    device
):
    """
    Re-run each trajectory in 'batch_action_seqs' from scratch with the current model,
    returning final tableaus, forward flows, backward flows, to pass to your cost/ loss.
    """
    replay_samples = []
    replay_fwd = []
    replay_bwd = []

    for acts in batch_action_seqs:
        tab, ff, bf = run_actions_with_current_model(
            acts,
            n_qubits,
            pf_model,
            pb_model,
            action_mapping,
            device
        )
        replay_samples.append(tab)
        replay_fwd.append(ff)
        replay_bwd.append(bf)
    return replay_samples, replay_fwd, replay_bwd

def load_checkpoint(checkpoint_filename, input_dim, hidden_dim, num_hidden_layers, output_dim, device=None):
    """
    Loads the training checkpoint from the given file.

    Parameters:
        checkpoint_filename (str): Path to the checkpoint file (.pth).
        n_qubits (int): Number of qubits, used to initialize the EquivariantHeisenbergNet.
        hidden_dim (int): Hidden dimension for the network.
        num_hidden_layers (int): Number of hidden layers for the network.
        output_dim (int): Output dimension (number of actions) for the network.
        device (torch.device, optional): Device to map the checkpoint to.
            Defaults to CUDA if available, otherwise CPU.

    Returns:
        pf_model: The loaded EquivariantHeisenbergNet.
        pb_model: The loaded DiscreteUniform model.
        opt: The optimizer with its state loaded.
        update: The update step stored in the checkpoint.
        tb_losses: Loss history from the checkpoint.
        top_batches: Top mini-batch information from the checkpoint.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct the models using the same architecture/hyperparameters as in training.
    pf_model = MLP(input_dim, hidden_dim, num_hidden_layers, output_dim).to(device)
    pb_model = DiscreteUniform(output_dim).to(device)
    
    # Recreate the optimizer for pf_model (ensure hyperparameters match training)
    opt = torch.optim.Adam(pf_model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Load the checkpoint (mapping to the appropriate device)
    checkpoint = torch.load(checkpoint_filename, map_location=device)

    # Load state dictionaries for the models and optimizer.
    pf_model.load_state_dict(checkpoint["pf_model_state_dict"])
    pb_model.load_state_dict(checkpoint["pb_model_state_dict"])
    opt.load_state_dict(checkpoint["optimizer_state_dict"])

    # Retrieve additional training metadata.
    update = checkpoint.get("update", None)
    tb_losses = checkpoint.get("tb_losses", None)
    top_batches = checkpoint.get("top_batches", None)

    print(f"Checkpoint loaded from '{checkpoint_filename}' at update {update}.")
    return pf_model, pb_model, opt, update, tb_losses, top_batches


def train_batch_replay(step=0,model_dir = "models_bell", pauli_str_list = ["II", "XX", "YY", "ZZ"],w_list = [0.25, 0.25, 0.25, 0.25],device=None):
    # -----------------------------------------------------
    # 1) Setup
    # -----------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Models saving directory
    os.makedirs(model_dir, exist_ok=True)
    
    w_list = torch.tensor(w_list,dtype=torch.float64,device=device)

    beta = 1.0
    epsilon = 1.0
    p = 100.0
    
    # Build models
    n_qubits = len(pauli_str_list[0])
    input_dim = (2 * n_qubits) * (2 * n_qubits + 1)
    hidden_dim = 512
    num_hidden_layers = 2
    lr = 1e-3
    weight_decay = 1e-4
    
    action_map = action_mapping(n_qubits)
    output_dim = len(action_map)
    
    #pf_model = EquivariantHeisenbergNet(n_qubits, hidden_dim, num_hidden_layers, output_dim)
    pf_model = MLP(input_dim, hidden_dim, num_hidden_layers, output_dim)
    pb_model = DiscreteUniform(output_dim)
    
    pf_model.to(device)
    pb_model.to(device)
    
    opt = torch.optim.Adam(pf_model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training hyperparams
    n_measurement = 100
    max_layer = 5
    max_episodes = 100000
    update_freq = 4
    num_updates = max_episodes // update_freq
    
    # Offset cost (move to GPU once)
    samples_opt = [CliffordTableau(n_qubits) for _ in range(n_measurement)]
    off_val, wsum = offset_cost(
        samples_opt, w_list, 
        epsilon=torch.tensor(epsilon, dtype=torch.float64, device=device)
    )

    print("offset:", off_val.item(), " wsum:", wsum.item())

    tb_losses = []
    K = 5  # keep top 5 mini-batches
    # We'll store top batches in a min-heap (reward, batch_seq) so
    # we can keep the highest rewards easily.
    top_batches = []

    # Save all the hyperparameters into a json file
    hyperparameters = {
        "n_qubits": n_qubits,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "output_dim": output_dim,
        "n_measurement": n_measurement,
        "max_layer": max_layer,
        "max_episodes": max_episodes,
        "update_freq": update_freq,
        "num_updates": num_updates,
        "beta": beta,
        "epsilon": epsilon,
        "p": p,
        "off_val": off_val.item(),
        "wsum": wsum.item(),
        "K": K,
        "optimizer": "Adam",
        "lr": lr,
        "weight_decay": weight_decay
    }
    with open(os.path.join(model_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=4)
        
    # Load checkpoint if needed
    checkpoint_filename = f"{model_dir}/checkpoint_H2_step_{step}.pth"
    update_start = step
    if os.path.exists(checkpoint_filename):
        print(f"Loading checkpoint from {checkpoint_filename}")
        pf_model, pb_model, opt, update_start, tb_losses, top_batches = load_checkpoint(
            checkpoint_filename, input_dim, hidden_dim, num_hidden_layers, output_dim, device
        )
    else:
        print(f"Checkpoint not found. Starting from scratch.")
        update_start = 0
        tb_losses = []
        top_batches = []

    # -----------------------------------------------------
    # 2) Training Loop
    # -----------------------------------------------------
    for update in range(update_start, num_updates):
        print(f"\n=== UPDATE {update+1}/{num_updates} ===")
        
        # -----------------
        # 2.1) Sample a big batch
        # -----------------
        total_samples = n_measurement * update_freq
        all_samples, all_fwd_flows, all_bwd_flows, all_action_seqs = sample_clifford_trajectories(
            total_samples, max_layer, n_qubits, pf_model, pb_model, action_map, device
        )
        
        # -----------------
        # 2.2) Compute cost ONCE for the entire big batch
        # -----------------
        # Vectorized cost for all trajectories
        #batch_cost_all =[cal_cost([i], pauli_str_list, w_list, epsilon=epsilon) for i in all_samples]
        #x_batch_all = (batch_cost_all - off_val) / (wsum - off_val + 1e-12)
        #batch_reward_all = beta * torch.exp(-p * x_batch_all)
        
        # We'll accumulate the total loss over sub-batches and then do one optimizer step
        total_loss = 0.0
        
        # -----------------
        # 2.3) Split into sub-batches
        # -----------------
        for i in range(update_freq):
            start_idx = i * n_measurement
            end_idx = (i + 1) * n_measurement

            batch_samples = all_samples[start_idx:end_idx]
            batch_fwd = torch.tensor(all_fwd_flows[start_idx:end_idx], dtype=torch.float32, device=device)
            batch_bwd = torch.tensor(all_bwd_flows[start_idx:end_idx], dtype=torch.float32, device=device)
            
            # Sub-batch cost or reward
            batch_cost = cal_cost(batch_samples, pauli_str_list, w_list, epsilon=epsilon)
            x_batch = (batch_cost - off_val) / (wsum - off_val + 1e-12)
            batch_reward = beta * torch.exp(-p * x_batch)
            # We only need to compute loss on sub-batch
            sub_loss = (pf_model.logZ + batch_fwd.sum() - batch_reward - batch_bwd.sum()).pow(2)
            total_loss += sub_loss
            print(f"Sub-batch {i+1}, reward: {batch_reward.item():.6f}, cost: {batch_cost.item():.6f}, offset: {off_val.item():.6f}, Loss: {sub_loss.item():.6f}")

            # Keep track of top mini-batches in a min-heap
            # We'll store the *average* reward for the mini-batch
            avg_reward = batch_reward #torch.mean(batch_reward).item()
            # If we have fewer than K items, just push
            if len(top_batches) < K:
                heapq.heappush(top_batches, (avg_reward, all_action_seqs[start_idx:end_idx]))
            else:
                # If the smallest reward in top_batches is < avg_reward, replace
                if top_batches[0][0] < avg_reward:
                    heapq.heapreplace(top_batches, (avg_reward, all_action_seqs[start_idx:end_idx]))

            # Print empirical averages for debugging
            print_reward_4qubit(batch_samples, i, pauli_str_list, w_list)
        
        # -----------------
        # 2.4) Single optimizer step for the entire big batch
        # -----------------
        avg_loss = total_loss / update_freq
        tb_losses.append(avg_loss.item())
        
        opt.zero_grad()
        avg_loss.backward()
        opt.step()

        print(f"Update step {update+1}, Loss: {avg_loss.item():.6f}, logZ: {pf_model.logZ.item():.3f}")

        # -----------------
        # 2.5) Replay every 10 updates
        # -----------------
        if (update + 1) % 10 == 0 and len(top_batches) > 0:
            print("\n** Replay top mini-batches **")
            replay_loss = 0.0
            
            for (rval, stored_action_seqs) in top_batches:
                # Re-run each top mini-batch with the current model => new flows
                replay_samples, replay_fwd, replay_bwd = replay_batch_with_current_model(
                    stored_action_seqs,  
                    n_qubits,
                    pf_model,
                    pb_model,
                    action_map,
                    device
                )
                # Now compute the new loss
                replay_cost = cal_cost(replay_samples, pauli_str_list, w_list, epsilon=epsilon)
                x_replay = (replay_cost - off_val) / (wsum - off_val + 1e-12)
                replay_reward = beta * torch.exp(-p * x_replay)
                #convert to tensors
                replay_fwd = torch.tensor(replay_fwd, dtype=torch.float32, device=device)
                replay_bwd = torch.tensor(replay_bwd, dtype=torch.float32, device=device)
                # Compute the loss for this mini-batch
                rl = (pf_model.logZ + torch.sum(replay_fwd) - replay_reward - torch.sum(replay_bwd)).pow(2)
                replay_loss += rl
                print(f"Replay step: reward:{replay_reward.item():.6f},cost: {replay_cost.item():.6f}, offset: {off_val.item():.6f}, Loss: {rl.item():.6f}")

            replay_loss = replay_loss / len(top_batches)
            opt.zero_grad()
            replay_loss.backward()
            opt.step()
            print(f"Replay Loss: {replay_loss.item():.6f}")

        # -----------------
        # 2.6) Save checkpoint periodically
        # -----------------
        if (update + 1) % 25 == 0:
            checkpoint = {
                'pf_model_state_dict': pf_model.state_dict(),
                'pb_model_state_dict': pb_model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'update': update + 1,
                'tb_losses': tb_losses,
                'top_batches': top_batches
            }
            checkpoint_filename = os.path.join(model_dir, f"checkpoint_H2_step_{update+1}.pth")
            torch.save(checkpoint, checkpoint_filename)
            print(f"Checkpoint saved: {checkpoint_filename}")
    
    print("Training complete.")
    return pf_model, pb_model, top_batches, tb_losses