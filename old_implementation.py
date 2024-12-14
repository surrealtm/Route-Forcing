import numpy as np
import cairo
from itertools import combinations

class QuantumCircuit:
    def __init__(self, gates, qubits):
        self.gates = gates
        self.qubits = qubits

    def get_executable_gates(self, placement):
        return [
            gate for gate in self.gates if all(qubit in placement for qubit in gate[1])
        ]

    def remove(self, gate):
        self.gates.remove(gate)

    def is_empty(self):
        return len(self.gates) == 0

    def required_interactions(self):
        return [(gate[1][0], gate[1][1]) for gate in self.gates if len(gate[1]) == 2]


class Topology:
    def __init__(self, connections):
        self.connections = connections

    def edges(self):
        edges = []
        for qubit, neighbors in self.connections.items():
            for neighbor in neighbors:
                if (neighbor, qubit) not in edges:
                    edges.append((qubit, neighbor))
        return edges

    def neighbors(self, qubit):
        return self.connections.get(qubit, [])
    

# ----------------------------
# Step 1: Partitioning
# ----------------------------

def generate_non_isomorphic_sub_architectures(topology, n):
    sub_architectures = []
    nodes = list(topology.connections.keys())
    for sub_nodes in combinations(nodes, n):
        sub_edges = [
            (u, v) for u, v in combinations(sub_nodes, 2) if v in topology.neighbors(u)
        ]
        sub_architectures.append((sub_nodes, sub_edges))
    return sub_architectures


def evaluate_sub_architecture(sub_arch):
    _, edges = sub_arch
    return len(edges)


def greedy_select_candidates(sub_architectures, num_parts):
    scores = [evaluate_sub_architecture(arch) for arch in sub_architectures]
    ranked = sorted(zip(sub_architectures, scores), key=lambda x: -x[1])
    selected = [arch for arch, _ in ranked[:num_parts]]
    return selected


def partition_circuit(circuit, sub_architectures):
    sub_circuits = []
    for sub_arch in sub_architectures:
        sub_nodes, _ = sub_arch
        sub_gates = [gate for gate in circuit.gates if set(gate[1]).issubset(sub_nodes)]
        sub_circuits.append(QuantumCircuit(sub_gates, sub_nodes))
    return sub_circuits


# ----------------------------
# Step 2: Mapping (Route-Forcing)
# ----------------------------

def initial_placement(sub_arch, sub_circuit):
    logical_qubits = sub_circuit.qubits
    physical_qubits = sub_arch[0]
    placement = {q: p for q, p in zip(logical_qubits, physical_qubits)}
    return placement


def calculate_swap_coefficients(sub_arch, sub_circuit, placement):
    swap_coefficients = {}
    for qubit_a, qubit_b in sub_circuit.required_interactions():
        force = np.array(placement[qubit_b]) - np.array(placement[qubit_a])
        for edge in sub_arch[1]:
            vector = np.array(edge[1]) - np.array(edge[0])
            swap_coefficients[edge] = np.dot(force, vector)
    return swap_coefficients


def select_swaps(swap_coefficients):
    selected_swaps = sorted(swap_coefficients.items(), key=lambda x: -x[1])[:1]
    return selected_swaps


def apply_swaps(swaps, placement, QC):
    for (qubit_a, qubit_b), _ in swaps:
        placement[qubit_a], placement[qubit_b] = placement[qubit_b], placement[qubit_a]
        QC.append(("SWAP", [ qubit_a, qubit_b ]))
        

def route_forcing(sub_arch, sub_circuit):
    QC = []
    placement = initial_placement(sub_arch, sub_circuit)
    
    while not sub_circuit.is_empty():
        executable_gates = sub_circuit.get_executable_gates(placement)
        for gate in executable_gates:
            QC.append(gate)
            sub_circuit.remove(gate)
        
        swap_coefficients = calculate_swap_coefficients(sub_arch, sub_circuit, placement)
        if swap_coefficients:
            swaps = select_swaps(swap_coefficients)
            apply_swaps(swaps, placement, QC)
    
    return QC


# ----------------------------
# Step 3: Integration
# ----------------------------

def integrate_sub_architectures(mapped_sub_circuits, interconnects):
    global_circuit = []
    for sub_circuit in mapped_sub_circuits:
        global_circuit.extend(sub_circuit)
    return global_circuit


# ----------------------------
# Main Algorithm
# ----------------------------

def sub_architecture_based_route_forcing(circuit, topology, sub_arch_size):
    sub_architectures = generate_non_isomorphic_sub_architectures(topology, sub_arch_size)
    best_sub_architectures = greedy_select_candidates(sub_architectures, len(circuit.qubits))
    sub_circuits = partition_circuit(circuit, best_sub_architectures)

    mapped_sub_circuits = []
    for sub_arch, sub_circuit in zip(best_sub_architectures, sub_circuits):
        mapped_sub_circuits.append(route_forcing(sub_arch, sub_circuit))

    global_circuit = integrate_sub_architectures(mapped_sub_circuits, topology.edges())
    return global_circuit

def default_route_forcing(circuit, topology):
    architecture = (list(topology.connections.keys()), topology.connections)
    return route_forcing(architecture, circuit)


# ----------------------------
# Visualization
# ----------------------------

def topology_qubit_position(topology: Topology, index):
    n = max(3, int(np.sqrt(len(topology.connections))))

    qubit_offset = 0.02
    qubit_gap    = (0.45 - qubit_offset * 2) / (n - 1)

    i = index % n
    j = index // n
    return qubit_offset + i * qubit_gap, qubit_offset + j * qubit_gap

def circuit_qubit_position(topology: Topology, circuit, qubit_index, gate_index):
    vertical_qubit_offset = 0.02
    vertical_qubit_gap    = (0.45 - vertical_qubit_offset * 2) / len(topology.connections)

    horizontal_gate_offset = 0.05
    horizontal_gate_gap    = (2 - horizontal_gate_offset * 2) / (len(circuit) + 1)
    
    return horizontal_gate_offset + gate_index * horizontal_gate_gap, 0.55 + vertical_qubit_offset + qubit_index * vertical_qubit_gap

def draw_qubit(context, x, y, index):
    qubit_radius = 0.01

    context.set_source_rgb(0, 0.5, 0.8)
    context.set_line_width(qubit_radius)
    context.arc(x, y, qubit_radius / 2, 0, 2 * np.pi)
    context.stroke()

    context.move_to(x, y)
    context.set_font_size(qubit_radius * 2)
    context.set_source_rgb(0, 0, 0)
    context.show_text("q" + str(index))
    context.stroke()

def draw(topology: Topology, circuit: [], output_name):
    height_in_points = 1024

    surface = cairo.SVGSurface(output_name + ".svg", height_in_points * 2, height_in_points)
    context = cairo.Context(surface)
    context.save()
    
    context.scale(height_in_points, height_in_points)
    context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)

    #
    # Draw a background
    #
    context.set_source_rgb(1, 1, 1)
    context.rectangle(0, 0, 2, 1)
    context.fill()
    context.stroke()
    
    #
    # Draw the topology
    #
    
    # Draw all edges between qubits
    for edge in topology.edges():
        fx, fy = topology_qubit_position(topology, edge[0])
        tx, ty = topology_qubit_position(topology, edge[1])

        context.set_source_rgb(0.3, 0.3, 0.3)
        context.set_line_width(0.005)
        context.move_to(fx, fy)
        context.line_to(tx, ty)
        context.stroke()

    # Draw all qubits
    for i in range(0, len(topology.connections)):
        x, y = topology_qubit_position(topology, i)
        draw_qubit(context, x, y, i)
        
        
    #
    # Draw the circuit
    #

    # Draw all qubits
    for i in range(0, len(topology.connections)):
        qx, qy = circuit_qubit_position(topology, circuit, i, 0)
        ex, ey = circuit_qubit_position(topology, circuit, i, len(circuit) + 1)

        context.set_source_rgb(0.3, 0.3, 0.3)
        context.set_line_width(0.005)
        context.move_to(qx, qy)
        context.line_to(ex, ey)
        context.stroke()

        draw_qubit(context, qx, qy, i)

    # Draw all gates
    for i in range(0, len(circuit)):
        gate = circuit[i]

        # Draw a text above the gate indicating the name
        x, y = circuit_qubit_position(topology, circuit, -1, i + 1)
        _, _, text_width, _, _, _ = context.text_extents(gate[0])
        context.set_source_rgb(0, 0, 0)
        context.set_font_size(0.02)
        context.move_to(x - text_width / 2, y)
        context.show_text(gate[0])
        context.stroke()

        # Draw a point on the qubits that are affected by this gate
        for j in range(0, len(gate[1])):
            qubit_radius = 0.01
            x, y = circuit_qubit_position(topology, circuit, gate[1][j], i + 1)            
            context.set_source_rgb(0.8, 0.5, 0.2)
            context.set_line_width(qubit_radius)
            context.arc(x, y, qubit_radius / 2, 0, 2 * np.pi)
            context.stroke()

            if j + 1 < len(gate[1]):
                ex, ey = circuit_qubit_position(topology, circuit, gate[1][j + 1], i + 1)
                context.set_line_width(0.005)
                context.move_to(x, y)
                context.line_to(ex, ey)
                context.stroke()
        
    context.restore()
    context.show_page()
    surface.finish()
    


# ----------------------------
# Example Execution
# ----------------------------

def execute_comparison(topology, circuit, name):
    with_subarchs = sub_architecture_based_route_forcing(circuit, topology, sub_arch_size = 3)
    without_subarchs = default_route_forcing(circuit, topology)
    draw(topology, with_subarchs, name + "_with_subarchs")
    draw(topology, without_subarchs, name + "_without_subarchs")

def three_topology():
    topology = Topology({
        0: [1],
        1: [0, 2],
        2: [1],
    })

    circuit = QuantumCircuit([
        ("CNOT", [0, 2])
    ], [0, 1, 2])

    execute_comparison(topology, circuit, "three")
    
def quad_topology():
    topology = Topology({
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4, 6],
        4: [1, 3, 5, 7],
        5: [2, 4, 8],
        6: [3, 7],
        7: [4, 6, 8],
        8: [5, 7],
    })

    circuit = QuantumCircuit([
        ("H", [0]),
        ("CNOT", [0, 1]),
        ("CNOT", [1, 2]),
        ("H", [3]),
        ("CNOT", [3, 4]),
        ("CNOT", [4, 5]),
        ("X", [6]),
        ("CNOT", [6, 7]),
        ("CNOT", [7, 8]),
        ("H", [2]),
        ("CNOT", [2, 5]),
        ("CNOT", [1, 4]),
    ], [0, 1, 2, 3, 4, 5, 6, 7, 8])

    execute_comparison(topology, circuit, "quad_topology")
    
if __name__ == "__main__":
    three_topology()
    #quad_topology()
