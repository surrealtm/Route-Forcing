import numpy as np
import cairo
from itertools import combinations



# --------------------------------------------- Data Structures ---------------------------------------------

Qubit   = int
Mapping = dict[Qubit, Qubit]
Edge    = tuple[Qubit, Qubit]
DAG     = dict[int, list['Gate']]

class Topology:
    connections: dict[Qubit, list[Qubit]]

    def __init__(self, connections: dict[Qubit, list[Qubit]]):
        self.connections = connections

    def connection_exists(self, q0: Qubit, q1: Qubit) -> bool:
        return q1 in self.connections[q0]

    def edges(self) -> list[Edge]:
        result: list[Edge] = []

        for q0 in self.connections.keys():
            for q1 in self.connections[q0]:
                edge0: Edge = (q0, q1)
                edge1: Edge = (q1, q0)
                if not edge0 in result and not edge1 in result:
                    result.append(edge0)
        
        return result
    
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

    def count_dependency_level(self) -> int:
        level = 0

        for dep in self.dependencies:
            deplevel = dep.count_dependency_level()
            if deplevel + 1 > level:
                level = deplevel + 1
        
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
            if gate.has_been_executed or not gate.all_dependencies_executed:
                continue

            executable = False
            
            if len(gate.operands) == 1:
                executable = True
            elif len(gate.operands) == 2:
                executable = topology.connection_exists(mapping[gate.operands[0]], mapping[gate.operands[1]])
            else:
                print("A gate was expected to have either 1 or 2 operands, not '" + str(len(gate.operands)) + "'!", file=sys.stderr)

            if exectuable:
                result.append(gate)
        
        return result

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
    
    def depth(self) -> int:
        return len(self.gates)

    def build_dag_for_drawing(self) -> DAG:
        result: DAG = {}

        for gate in self.gates:
            depth = gate.count_dependency_level()

            if not depth in result:
                result[depth] = []
                
            result[depth].append(gate)

        return result
    


# ---------------------------------------------- Route-Forcing ----------------------------------------------

def route_forcing(circuit: Circuit, topology: Topology) -> Circuit:
    result: Circuit = Circuit([], [])

    circuit.reset_execution_state()
    circuit.build_dependencies()

    #
    # Iterate while there are still some unexected gates left
    #
    # while circuit.count_unexecuted_gates() > 0:

    return result
    


# ---------------------------------------------- Visualization ----------------------------------------------

def topology_qubit_position(topology: Topology, index: int) -> (float, float):
    n = max(3, int(np.sqrt(len(topology.connections))))

    qubit_offset = 0.02
    qubit_gap    = (0.45 - qubit_offset * 2) / (n - 1)

    i = index % n
    j = index // n
    return qubit_offset + i * qubit_gap, qubit_offset + j * qubit_gap

def circuit_qubit_position(topology: Topology, circuit: Circuit, qubit_index: int, gate_index: int) -> (float, float):
    vertical_qubit_offset = 0.02
    vertical_qubit_gap    = (0.45 - vertical_qubit_offset * 2) / len(topology.connections)

    horizontal_gate_offset = 0.05
    horizontal_gate_gap    = (2 - horizontal_gate_offset * 2) / (circuit.depth() + 1)
    
    return horizontal_gate_offset + gate_index * horizontal_gate_gap, 0.55 + vertical_qubit_offset + qubit_index * vertical_qubit_gap

def gate_position_in_dag(dag: DAG, gate: Gate) -> (float, float):
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
    context.arc(x, y, qubit_radius / 2, 0, 2 * np.pi)
    context.stroke()

    context.move_to(x, y)
    context.set_font_size(qubit_radius * 2)
    context.set_source_rgb(0, 0, 0)
    context.show_text("q" + str(index))
    context.stroke()

def draw(topology: Topology, circuit: Circuit, output_name: str):
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
    # Draw the DAG
    #

    dag = circuit.build_dag_for_drawing()

    # Draw all connections between gates
    for depth in dag:
        for gate in dag[depth]:
            for dep in gate.dependencies:
                gate_x, gate_y = gate_position_in_dag(dag, gate)
                dep_x, dep_y   = gate_position_in_dag(dag, dep)

                context.set_source_rgb(0.3, 0.3, 0.3)
                context.set_line_width(0.005)
                context.move_to(gate_x, gate_y)
                context.line_to(dep_x, dep_y)
                context.stroke()

    # Draw all gates
    for depth in dag:
        for gate in dag[depth]:
            gate_radius = 0.05
            x, y = gate_position_in_dag(dag, gate)

            context.set_source_rgb(0.8, 0.5, 0.2)
            context.set_line_width(gate_radius)
            context.arc(x, y, gate_radius / 2, 0, 2 * np.pi)
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
    for i in range(0, len(topology.connections)):
        qx, qy = circuit_qubit_position(topology, circuit, i + 1, 0)
        ex, ey = circuit_qubit_position(topology, circuit, i + 1, circuit.depth() + 1)

        context.set_source_rgb(0.3, 0.3, 0.3)
        context.set_line_width(0.005)
        context.move_to(qx, qy)
        context.line_to(ex, ey)
        context.stroke()

        draw_qubit(context, qx, qy, i)
        
    # Draw all gates
    for i in range(0, circuit.depth()):
        gate = circuit.gates[i]

        # Draw a text above the gate indicating the name
        x, y = circuit_qubit_position(topology, circuit, 0, i + 1)
        _, _, text_width, _, _, _ = context.text_extents(gate.name)
        context.set_source_rgb(0, 0, 0)
        context.set_font_size(0.02)
        context.move_to(x - text_width / 2, y)
        context.show_text(gate.name)
        context.stroke()

        # Draw a point on the qubits that are affected by this gate
        for j in range(0, len(gate.operands)):
            qubit_radius = 0.01
            x, y = circuit_qubit_position(topology, circuit, gate.operands[j] + 1, i + 2)
            context.set_source_rgb(0.8, 0.5, 0.2)
            context.set_line_width(qubit_radius)
            context.arc(x, y, qubit_radius / 2, 0, 2 * np.pi)
            context.stroke()

            # Draw an edge to the next qubit affected by this gate
            if j + 1 < len(gate.operands):
                ex, ey = circuit_qubit_position(topology, circuit, gate.operands[j + 1] + 1, i + 2)
                context.set_line_width(0.005)
                context.move_to(x, y)
                context.line_to(ex, ey)
                context.stroke()
        
    context.restore()
    context.show_page()
    surface.finish()



# ------------------------------------------- Testing Entry Point -------------------------------------------

def execute_comparison(topology: Topology, circuit: Circuit, name: str):
    # with_subarchs = sub_architecture_based_route_forcing(circuit, topology, sub_arch_size = 3)
    # draw(topology, with_subarchs, name + "_with_subarchs")

    without_subarchs = route_forcing(circuit, topology)
    draw(topology, without_subarchs, name + "_without_subarchs")

def three_topology():
    topology = Topology({
        0: [1],
        1: [0, 2],
        2: [1],
    })

    circuit = Circuit(
        [ 0, 1, 2 ],
        [ Gate("CNOT", [0, 2]) ]
    )

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

    circuit = Circuit(
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8 ],
        [ Gate("H",    [0]),
          Gate("CNOT", [0, 1]),
          Gate("CNOT", [1, 2]),
          Gate("H",    [3]),
          Gate("CNOT", [3, 4]),
          Gate("CNOT", [4, 5]),
          Gate("X",    [6]),
          Gate("CNOT", [6, 7]),
          Gate("CNOT", [7, 8]),
          Gate("H",    [2]),
          Gate("CNOT", [2, 5]),
          Gate("CNOT", [1, 4]) ]
    )

    execute_comparison(topology, circuit, "quad_topology")
    
if __name__ == "__main__":
    #three_topology()
    quad_topology()
