import cairo
import time
import math
import random



# ------------------------------------------------- Helpers -------------------------------------------------

DEBUG_LOG   = True
DEBUG_SLEEP = True

RANDOMIZE_SWAPS = False # This doesn't fix anything deterministically...

def log(format, *args):
    if DEBUG_LOG:
        print(format, *args)


# --------------------------------------------- Data Structures ---------------------------------------------

Qubit    = int
Mapping  = dict[Qubit, Qubit]
Edge     = tuple[Qubit, Qubit]
DAG      = dict[int, list['Gate']]
Vector2  = tuple[float, float]


def get_initial_mapping(virtual_qubits: list[Qubit], physical_qubits: list[Qubit]) -> Mapping:
    mapping: Mapping = {}

    assert(len(virtual_qubits) <= len(physical_qubits))
    
    for i in range(0, len(virtual_qubits)):
        mapping[virtual_qubits[i]] = physical_qubits[i]

    return mapping

def update_mapping(mapping: Mapping, virtual: Qubit, physical: Qubit):
    mapping[virtual] = physical

def query_mapping_inverse(mapping: Mapping, physical: Qubit) -> Qubit:
    for virtual in mapping.keys():
        if mapping[virtual] == physical:
            return virtual

    return -1


class Topology:
    connections: dict[Qubit, list[Qubit]]

    def __init__(self, connections: dict[Qubit, list[Qubit]]):
        self.connections = connections

    def connection_exists(self, q0: Qubit, q1: Qubit) -> bool:
        return q1 in self.connections[q0]

    def get_edges(self) -> list[Edge]:
        result: list[Edge] = []

        for q0 in self.connections.keys():
            for q1 in self.connections[q0]:
                edge0: Edge = (q0, q1)
                edge1: Edge = (q1, q0)

                if (not edge0 in result and not edge1 in result):
                    result.append(edge0)
        
        return result

    def get_edges_with_outgoing_qubit(self, qubit: Qubit) -> list[Edge]:
        result: list[Edge] = []

        for q0 in self.connections.keys():
            for q1 in self.connections[q0]:
                edge0: Edge = (q0, q1)
                edge1: Edge = (q1, q0)

                if edge0[0] == qubit and not edge0 in result:
                    result.append(edge0)

                if edge1[0] == qubit and not edge1 in result:
                    result.append(edge1)
        
        return result
    
    def get_qubits(self) -> int:
        return list(self.connections.keys())

    def get_qubit_count(self) -> int:
        return len(self.connections.keys())

    def qubit_index(self, qubit: Qubit) -> int:
        keys = list(self.connections.keys())
        return keys.index(qubit)

    def get_topology_row_count(self):
        qubit_count = max(self.get_qubits()) + 1
        return max(3, int(math.sqrt(qubit_count)))

    def get_circuit_row_count(self):
        qubit_count = max(self.get_qubits()) + 1
        return qubit_count
    
    def get_physical_position(self, qubit) -> Vector2:
        row_count = self.get_topology_row_count()
        index     = qubit
        column    = index // row_count
        row       = index % row_count
        return (column, row)

    def calculate_physical_delta(self, q0: Qubit, q1: Qubit) -> Vector2:
        p0 = self.get_physical_position(q0)
        p1 = self.get_physical_position(q1)
        direction = (p1[0] - p0[0], p1[1] - p0[1])
        magnitude = math.sqrt(direction[0] * direction[0] + direction[1] * direction[1])
        return (direction[0] / magnitude, direction[1] / magnitude)
    
class SWAP:
    edge: Edge
    coefficient: float

    def __init__(self, edge: Edge, coefficient: float):
        self.edge = edge
        self.coefficient = coefficient

    def __str__(self):
        return "[ " + str(self.edge[0]) + " <-> " + str(self.edge[1]) + " (" + str(self.coefficient) + ") ]"

    def __repr__(self):
        return "SWAP(" + str(self.edge) + ", " + str(self.coefficient) + ")"
        
    def is_executable(self, topology: Topology, mapping: Mapping, swapped_qubits: list[Qubit]) -> bool:
        # The 'self.coefficient > 0' part is not explicitly mentioned in the paper, but it otherwise we often
        # get stuck in an infinite loop...
        return self.coefficient > 0 and not self.edge[0] in swapped_qubits and not self.edge[1] in swapped_qubits and topology.connection_exists(self.edge[0], self.edge[1])

    def execute(self, topology: Topology, mapping: Mapping, swapped_qubits: list[Qubit], circuit: 'Circuit'):
        virtual0  = query_mapping_inverse(mapping, self.edge[0])
        virtual1  = query_mapping_inverse(mapping, self.edge[1])
        physical0 = self.edge[1]
        physical1 = self.edge[0]
        update_mapping(mapping, virtual0, physical0)
        update_mapping(mapping, virtual1, physical1)
        swapped_qubits.append(self.edge[0])
        swapped_qubits.append(self.edge[1])
        circuit.gates.append(Gate("SWAP", [ self.edge[0], self.edge[1] ]))
        log("Executing:", self)
    
class Gate:
    name: str                  = ""
    operands: list[Qubit]      = []
    dependencies: list['Gate'] = []
    has_been_executed: bool    = False
    
    def __init__(self, name: str, operands: list[Qubit]):
        self.name              = name
        self.operands          = operands
        self.dependencies      = []
        self.has_been_executed = False

    def all_dependencies_executed(self) -> bool:
        result = True

        for gate in self.dependencies:
            if not gate.has_been_executed:
                result = False
        
        return result

    def get_dag_level(self) -> int:
        level = 0

        for dep in self.dependencies:
            level = max(level, dep.get_dag_level() + 1)
        
        return level
    
# The Route-Forcing paper uses an actual Directed Acyclic Graph here. We do not create an explicit data structure
# for that here, instead we implicitly store it in the circuit:
# Each gate has a list of dependencies, which are the previous gates in the DAG. Each gate also stores whether
# it has been executed or not. Executed gates would be removed from the DAG, we however only ignore it when
# searching for executable gates.
class Circuit:
    qubits = list[Qubit]
    gates  = list[Gate]
    
    def __init__(self, qubits: list[Qubit], gates: list[Gate]):
        self.qubits = qubits
        self.gates  = gates
    
    def get_executable_gates(self, topology: Topology, mapping: Mapping) -> list[Gate]:
        result = []

        for gate in self.gates:
            if gate.has_been_executed or not gate.all_dependencies_executed():
                continue

            executable = False
            
            if len(gate.operands) == 1:
                executable = True
            elif len(gate.operands) == 2:
                executable = topology.connection_exists(mapping[gate.operands[0]], mapping[gate.operands[1]])
            else:
                print("A gate was expected to have either 1 or 2 operands, not '" + str(len(gate.operands)) + "'!", file=sys.stderr)

            if executable:
                result.append(gate)
        
        return result

    def get_next_executable_gate(self, topology: Topology, mapping: Mapping) -> (bool, Gate):
        # This is very wasteful but it works for now
        gates = self.get_executable_gates(topology, mapping)
        if len(gates) == 0:
            return False, Gate("---", [])

        return True, gates[0]

    def get_qubits(self) -> list[Qubit]:
        return self.qubits
    
    def count_unexecuted_gates(self) -> int:
        count = 0

        for gate in self.gates:
            if not gate.has_been_executed:
                count += 1
        
        return count
    
    def build_dependencies(self):
        latest_gate_for_qubit: dict[Qubit, Gate] = {}

        for gate in self.gates:
            for operand in gate.operands:
                if operand in latest_gate_for_qubit:
                    gate.dependencies.append(latest_gate_for_qubit[operand])
                latest_gate_for_qubit[operand] = gate
    
    def reset_execution_state(self):
        for gate in self.gates:
            gate.has_been_executed = False
    
    def get_depth(self) -> int:
        return len(self.gates)

    def get_dag_depth(self) -> int:
        level = 0

        for gate in self.gates:
            level = max(level, gate.get_dag_level())
        
        return level + 1

    def get_dag_layer(self, level: int) -> list[Gate]:
        result: list[Gate] = []

        for gate in self.gates:
            if level == gate.get_dag_level():
                result.append(gate)
        
        return result
    
    def build_dag_for_drawing(self) -> DAG:
        result: DAG = {}

        for gate in self.gates:
            level = gate.get_dag_level()

            if not level in result:
                result[level] = []
                
            result[level].append(gate)

        return result
    


# ---------------------------------------------- Route-Forcing ----------------------------------------------
    
def update_swap_coefficient(swaps: list[SWAP], edge: Edge, coefficient: float):
    # Check if the edge already exists in the list. We are treating edges as directed here, because the swap
    # coefficient is a dot product (and therefore dependent on the direction vector of the edge).
    for swap in swaps:
        if (swap.edge[0] == edge[0] and swap.edge[1] == edge[1]):
            swap.coefficient += coefficient
            return

    swaps.append(SWAP(edge, coefficient))

def calculate_swap_coefficient(topology: Topology, edge: Edge, q0: Qubit, q1: Qubit, temporal_distance: float) -> float:
    operand_delta = topology.calculate_physical_delta(q0, q1)
    edge_delta = topology.calculate_physical_delta(edge[0], edge[1])
    attraction_force = operand_delta[0] * edge_delta[0] + operand_delta[1] * edge_delta[1]
    log("     - Edge " + str(edge[0]) + " -> " + str(edge[1]) + " : " + str(attraction_force))
    return attraction_force * temporal_distance

def calculate_all_swap_coefficients(swaps: list[SWAP], topology: Topology, mapping: Mapping, q0: Qubit, q1: Qubit, temporal_distance: float):
    log(" > Swaps for gate " + str(q0) + " - " + str(q1) + ", mapped to " + str(mapping[q0]) + " - " + str(mapping[q1]))

    edges = topology.get_edges_with_outgoing_qubit(mapping[q0])
    for edge in edges:
        coefficient = calculate_swap_coefficient(topology, edge, mapping[q0], mapping[q1], temporal_distance)
        update_swap_coefficient(swaps, edge, coefficient)
        
def route_forcing(circuit: Circuit, topology: Topology) -> (Circuit, DAG):
    result: Circuit = Circuit([], [])

    mapping: Mapping = get_initial_mapping(circuit.get_qubits(), topology.get_qubits())
    
    circuit.reset_execution_state()
    circuit.build_dependencies()
    dag = circuit.build_dag_for_drawing()
    
    dag_depth = circuit.get_dag_depth()

    #
    # Iterate while there are still some unexected gates left
    #
    while circuit.count_unexecuted_gates() > 0:
        # Debug print the state
        log("=== Route-Forcing Step ===")
        log("Mapping:", mapping)

        # Execute all gates that can be executed right now
        found_executable_gate, executable_gate = circuit.get_next_executable_gate(topology, mapping)
        while found_executable_gate:
            executable_gate.has_been_executed = True
            result.gates.append(Gate(executable_gate.name, [ mapping[operand] for operand in executable_gate.operands ]))
            found_executable_gate, executable_gate = circuit.get_next_executable_gate(topology, mapping)

        # Look ahead in the DAG to assign a swap coefficient to all possible swaps
        swaps: list[SWAP] = []
        swapped_qubits: list[Qubit] = [] # Qubits that have been part of a SWAP operations no longer hold the value they are expected to, therefore we shouldn't do any more swaps on edges containing these qubits

        for i in range(0, dag_depth):
            gates = circuit.get_dag_layer(i)
            for gate in gates:
                if gate.has_been_executed or len(gate.operands) != 2:
                    continue

                temporal_distance = 1 / i
                calculate_all_swap_coefficients(swaps, topology, mapping, gate.operands[0], gate.operands[1], temporal_distance)
                calculate_all_swap_coefficients(swaps, topology, mapping, gate.operands[1], gate.operands[0], temporal_distance)
                        
        # Sort the possible swaps by their coefficient
        swaps.sort(reverse = True, key = lambda e: e.coefficient)

        # Improve the chance of ever converging by randomly exchanging swaps in the list. This is what the
        # paper proposes...
        if RANDOMIZE_SWAPS:
            randomness_factor = random.randrange(0, max(len(swaps) // 3, 1))
            for i in range(0, randomness_factor):
                my_edge    = random.randrange(0, len(swaps))
                other_edge = random.randrange(0, len(swaps))
                swaps[my_edge], swaps[other_edge] = swaps[other_edge], swaps[my_edge]
            
        # Executable all possible swaps in descending order
        for swap in swaps:
            if swap.is_executable(topology, mapping, swapped_qubits):
                swap.execute(topology, mapping, swapped_qubits, result)

        if DEBUG_SLEEP:
            time.sleep(0.5)
                
    return result, dag
    


# ---------------------------------------------- Visualization ----------------------------------------------

def get_topology_qubit_position(topology: Topology, qubit: Qubit) -> (float, float):
    qubit_offset = 0.02
    qubit_gap    = (0.45 - qubit_offset * 2) / (topology.get_topology_row_count() - 1)

    position = topology.get_physical_position(qubit)
    
    return qubit_offset + position[0] * qubit_gap, qubit_offset + position[1] * qubit_gap

def get_circuit_qubit_position(topology: Topology, circuit: Circuit, qubit_index: int, gate_index: int) -> (float, float):
    vertical_qubit_offset = 0.02
    vertical_qubit_gap    = (0.45 - vertical_qubit_offset * 2) / topology.get_circuit_row_count()

    horizontal_gate_offset = 0.05
    horizontal_gate_gap    = (2 - horizontal_gate_offset * 2) / (circuit.get_depth() + 1)
    
    return horizontal_gate_offset + gate_index * horizontal_gate_gap, 0.55 + vertical_qubit_offset + qubit_index * vertical_qubit_gap

def get_gate_position_in_dag(dag: DAG, gate: Gate) -> (float, float):
    depth = 0
    index_at_depth = 0
    gates_at_depth = 1

    dag_depth = len(dag.keys())
    
    for i in range(0, dag_depth):
        if gate in dag[i]:
            depth = i
            index_at_depth = dag[i].index(gate)
            gates_at_depth = len(dag[i])

    horizontal_offset = 1.1
    horizontal_gap    = 0.8 / dag_depth

    vertical_gap    = 0.4 / gates_at_depth
    vertical_offset = 0.05 + vertical_gap / 2
    
    return horizontal_offset + depth * horizontal_gap, vertical_offset + index_at_depth * vertical_gap
    
def draw_qubit(context: cairo.Context, x: float, y: float, index: int):
    qubit_radius = 0.01

    context.set_source_rgb(0, 0.5, 0.8)
    context.set_line_width(qubit_radius)
    context.arc(x, y, qubit_radius / 2, 0, 2 * math.pi)
    context.stroke()

    context.set_font_size(qubit_radius * 2)
    name = "q" + str(index)
    _, _, text_width, _, _, _ = context.text_extents(name)
    context.move_to(x - text_width / 2, y)
    context.set_source_rgb(0, 0, 0)
    context.show_text(name)
    context.stroke()

def draw(topology: Topology, dag: DAG, mapped_circuit: Circuit, output_name: str):
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
    # Subdivide the space into three parts, the topology, the DAG and the circuit
    #
    context.set_source_rgb(0, 0, 0)
    context.set_line_width(0.005)

    context.move_to(0, 0.5)
    context.line_to(2, 0.5)
    context.stroke()

    context.move_to(1, 0)
    context.line_to(1, 0.5)
    context.stroke()
        
    
    #
    # Draw the topology
    #
    
    # Draw all edges between qubits
    for edge in topology.get_edges():
        fx, fy = get_topology_qubit_position(topology, edge[0])
        tx, ty = get_topology_qubit_position(topology, edge[1])

        context.set_source_rgb(0.3, 0.3, 0.3)
        context.set_line_width(0.005)
        context.move_to(fx, fy)
        context.line_to(tx, ty)
        context.stroke()

    # Draw all qubits
    for qubit in topology.get_qubits():
        x, y = get_topology_qubit_position(topology, qubit)
        draw_qubit(context, x, y, qubit)


        
    #
    # Draw the DAG
    #

    # Draw all connections between gates
    for depth in dag:
        for gate in dag[depth]:
            for dep in gate.dependencies:
                gate_x, gate_y = get_gate_position_in_dag(dag, gate)
                dep_x, dep_y   = get_gate_position_in_dag(dag, dep)

                context.set_source_rgb(0.3, 0.3, 0.3)
                context.set_line_width(0.005)
                context.move_to(gate_x, gate_y)
                context.line_to(dep_x, dep_y)
                context.stroke()

    # Draw all gates
    for depth in dag:
        for gate in dag[depth]:
            gate_radius = 0.03
            x, y = get_gate_position_in_dag(dag, gate)

            context.set_source_rgb(0.8, 0.5, 0.2)
            context.set_line_width(gate_radius)
            context.arc(x, y, gate_radius / 2, 0, 2 * math.pi)
            context.stroke()

            _, _, text_width, _, _, _ = context.text_extents(gate.name)
            context.set_source_rgb(0, 0, 0)
            context.set_font_size(0.02)
            context.move_to(x - text_width / 2, y)
            context.show_text(gate.name)
            context.move_to(x - text_width / 2, y + 0.02)
            context.show_text(str(gate.operands))
            context.stroke()

            
            
    #
    # Draw the circuit
    #

    # Draw all qubits
    for qubit in topology.get_qubits():
        qx, qy = get_circuit_qubit_position(topology, mapped_circuit, qubit + 1, 0)
        ex, ey = get_circuit_qubit_position(topology, mapped_circuit, qubit + 1, mapped_circuit.get_depth() + 1)

        context.set_source_rgb(0.3, 0.3, 0.3)
        context.set_line_width(0.005)
        context.move_to(qx, qy)
        context.line_to(ex, ey)
        context.stroke()

        draw_qubit(context, qx, qy, qubit)
        
    # Draw all gates
    for i in range(0, mapped_circuit.get_depth()):
        gate = mapped_circuit.gates[i]

        # Draw a text above the gate indicating the name
        x, y = get_circuit_qubit_position(topology, mapped_circuit, 0, i + 1)
        _, _, text_width, _, _, _ = context.text_extents(gate.name)
        context.set_source_rgb(0, 0, 0)
        context.set_font_size(0.02)
        context.move_to(x - text_width / 2, y)
        context.show_text(gate.name)
        context.stroke()

        # Draw a point on the qubits that are affected by this gate
        for j in range(0, len(gate.operands)):
            qubit_radius = 0.01
            x, y = get_circuit_qubit_position(topology, mapped_circuit, gate.operands[j] + 1, i + 1)
            context.set_source_rgb(0.8, 0.5, 0.2)
            context.set_line_width(qubit_radius)
            context.arc(x, y, qubit_radius / 2, 0, 2 * math.pi)
            context.stroke()

            # Draw an edge to the next qubit affected by this gate
            if j + 1 < len(gate.operands):
                ex, ey = get_circuit_qubit_position(topology, mapped_circuit, gate.operands[j + 1] + 1, i + 1)
                context.set_line_width(0.005)
                context.move_to(x, y)
                context.line_to(ex, ey)
                context.stroke()
        
    context.restore()
    context.show_page()
    surface.finish()



# ------------------------------------------- Testing Entry Point -------------------------------------------

def execute_comparison(topology: Topology, circuit: Circuit, name: str):
    with_start = time.perf_counter()
    with_subarchs, dag = route_forcing(circuit, topology) # @Incomplete: Obviously
    with_end = time.perf_counter()
    draw(topology, dag, with_subarchs, "with_subarchs")
    print("With:    " + str(with_end - with_start) + "s.")

    without_start = time.perf_counter()
    without_subarchs, dag = route_forcing(circuit, topology)
    without_end = time.perf_counter()
    draw(topology, dag, without_subarchs, "without_subarchs")
    print("Without: " + str(without_end - without_start) + "s.")
    

def three_topology():
    topology = Topology({
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4],
        4: [1, 3, 5],
        5: [2, 4],
    })

    circuit = Circuit(
        topology.get_qubits(),
        [ Gate("CNOT", [0, 5]) ]
    )

    execute_comparison(topology, circuit, "three")
    
def quad_topology():
    topology = Topology({
        0: [1, 3],
        1: [0, 2],
        2: [1, 5],
        3: [0, 6],
        5: [2, 8],
        6: [3, 7, 9],
        7: [6, 8, 9],
        8: [5, 7],
        9: [6, 7],
    })
    
    circuit = Circuit(
        topology.get_qubits(),
        [
            Gate("H",    [1]),
            Gate("H",    [2]),
            Gate("CNOT", [1, 2]),

            Gate("H",    [0]),
            Gate("H",    [3]),
            Gate("CNOT", [0, 3]),

            Gate("H",    [6]),
            Gate("H",    [7]),
            Gate("CNOT", [6, 7]),            
            Gate("H",    [5]),
            Gate("H",    [8]),
            Gate("CNOT", [5, 8]),
            Gate("CNOT", [5, 6]),
            Gate("H",    [9]),
            Gate("CNOT", [6, 9]),
            Gate("CNOT", [7, 9]),
        ]
    )

    execute_comparison(topology, circuit, "quad_topology")
    
if __name__ == "__main__":
    #three_topology()
    quad_topology()
