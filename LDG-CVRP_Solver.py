import streamlit as st
import vrplib
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import os
import tempfile  # Per creare file temporanei

# Initialize session state variables if they don't exist
if 'solution_data' not in st.session_state:
    st.session_state.solution_data = None
if 'json_str' not in st.session_state:
    st.session_state.json_str = None
if 'solution_filename' not in st.session_state:
    st.session_state.solution_filename = None
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'manager' not in st.session_state:
    st.session_state.manager = None
if 'routing' not in st.session_state:
    st.session_state.routing = None

# Configurazione della pagina
st.set_page_config(layout="wide")

# Titolo della dashboard
st.markdown(
    """
    <h1 style='text-align: center; font-size: 48px;'>CVRP Solver - ORTools</h1>
    """,
    unsafe_allow_html=True
)

# Aggiunta dello stile
st.markdown(
    """
    <style>
    .column-title {
        font-size: 20px;
        font-weight: bold;
        color: #FFFFFF; /* Colore del titolo */
        padding: 10px;
        background-color: #23272A; /* Colore di sfondo del titolo */
        border-radius: 5px;
    }
    .placeholder {
        font-size: 16px;
        color: #ADD8E6; /* Colore del testo del placeholder */
        text-align: center;
        padding: 20px;
        border: 2px dashed #ADD8E6; /* Bordo tratteggiato per il placeholder */
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 20px; /* Imposta la dimensione del font desiderata */
    }
    /* Riduci la dimensione del font delle etichette nei widget st.metric */
    div[data-testid="stMetricLabel"] {
        font-size: 14px; /* Imposta la dimensione del font desiderata */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Creazione delle colonne
col1, col2, col3 = st.columns([1.05, 1.7, 0.8])

# Definizione delle funzioni (spostate qui per essere definite prima dell'uso)

# Funzione per calcolare la matrice delle distanze euclidee
def compute_euclidean_distance_matrix(locations, scale_factor):
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations), dtype=int)
    for from_idx in range(num_locations):
        for to_idx in range(num_locations):
            if from_idx != to_idx:
                from_node = locations[from_idx]
                to_node = locations[to_idx]
                distance = np.hypot(from_node[0] - to_node[0], from_node[1] - to_node[1])
                distance_matrix[from_idx][to_idx] = int(distance * scale_factor)
    return distance_matrix.tolist()

# Funzione per creare il data model
def create_data_model(instance, num_vehicles):
    data = {}
    data['locations'] = instance['node_coord']
    data['scale_factor'] = 1000
    data['distance_matrix'] = compute_euclidean_distance_matrix(data['locations'], data['scale_factor'])
    data['demands'] = instance['demand']
    data['vehicle_capacities'] = [instance['capacity']] * num_vehicles
    data['num_vehicles'] = num_vehicles
    data['depot'] = int(instance['depot'][0])
    return data

# Funzione per ottenere i parametri di ricerca
def get_search_parameters(heuristic_name, use_metaheuristic, metaheuristic_name, time_limit):
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.log_search = True

    heuristics = {
        'AUTOMATIC': routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
        'PATH_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        'PATH_MOST_CONSTRAINED_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        'EVALUATOR_STRATEGY': routing_enums_pb2.FirstSolutionStrategy.EVALUATOR_STRATEGY,
        'SAVINGS': routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        'SWEEP': routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        'CHRISTOFIDES': routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        'ALL_UNPERFORMED': routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        'BEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
        'PARALLEL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        'LOCAL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
        'GLOBAL_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
        'LOCAL_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
        'FIRST_UNBOUND_MIN_VALUE': routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE
    }

    metaheuristics = {
        'AUTOMATIC': routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
        'GREEDY_DESCENT': routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
        'GUIDED_LOCAL_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        'SIMULATED_ANNEALING': routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        'TABU_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        'GENERIC_TABU_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH
    }

    search_parameters.first_solution_strategy = heuristics.get(heuristic_name, routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)

    if use_metaheuristic:
        search_parameters.local_search_metaheuristic = metaheuristics.get(metaheuristic_name, routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    else:
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT

    search_parameters.time_limit.FromSeconds(time_limit)
    return search_parameters

# Funzione per ottenere i dati della soluzione
def get_solution_data(data, manager, routing, solution, scale_factor, instance_name, heuristic_name, metaheuristic_name, execution_time):
    total_distance = 0
    total_load = 0
    routes = {}
    objective = solution.ObjectiveValue() / scale_factor

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_load = 0
        route_nodes = []

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += int(data["demands"][node_index])
            route_nodes.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        route_distance += routing.GetArcCostForVehicle(index, routing.Start(vehicle_id), vehicle_id)
        depot_index = manager.IndexToNode(index)
        route_nodes.append(depot_index)

        total_distance += float(route_distance)
        total_load += int(route_load)

        routes[vehicle_id + 1] = {
            'route': route_nodes,
            'distance': float(route_distance) / scale_factor,
            'load': int(route_load)
        }

    solution_data = {
        'instance_name': instance_name,
        'objective': float(objective),
        'total_distance': float(total_distance) / scale_factor,
        'total_load': int(total_load),
        'execution_time': float(execution_time),
        'heuristic': heuristic_name,
        'metaheuristic': metaheuristic_name,
        'routes': routes
    }

    return solution_data

# Funzione per gestire i tipi non serializzabili
def default(o):
    if isinstance(o, np.integer):
        return int(o)
    elif isinstance(o, np.floating):
        return float(o)
    else:
        return o.__str__()

def plot_instance(instance):
    locations = instance['node_coord']
    depot_index = instance['depot'][0]
    locations_array = np.array(locations)
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(8, 4.8))
    ax = plt.gca()
    
    # Plot del deposito
    ax.scatter(
        locations_array[depot_index, 0],
        locations_array[depot_index, 1],
        color='red',
        marker='s',
        s=100,
        label='Deposito'
    )
    
    # Plot dei clienti
    customer_indices = list(range(len(locations)))
    customer_indices.remove(depot_index)
    ax.scatter(
        locations_array[customer_indices, 0],
        locations_array[customer_indices, 1],
        color='paleturquoise',
        marker='s',
        s=100,
        label='Clienti'
    )

    # Annotazioni dei nodi
    for i, (x, y) in enumerate(locations):
        if i == depot_index:
            # Colore per il deposito
            bbox_props = dict(facecolor='red', edgecolor='black', boxstyle='round,pad=0.2')
        else:
            # Colore per i clienti
            bbox_props = dict(facecolor='paleturquoise', edgecolor='black', boxstyle='round,pad=0.2')
        ax.annotate(
            str(i),
            (x, y),
            fontsize=8,
            ha='center',
            va='center',
            bbox=bbox_props
        )

    # Impostazioni degli assi e titolo
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title('Istanza CVRP', fontsize=14)
    
    # Aggiunta della legenda
    ax.legend(
        loc='best',
        fontsize=10,
        framealpha=1,
        facecolor='white',
        edgecolor='black',
        shadow=True
    )
    
    # Personalizzazione della griglia
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig

# Funzione per plottare la soluzione
def plot_solution(data, manager, routing, solution):
    locations = data['locations']
    depot_index = data['depot']
    colors = list(mcolors.TABLEAU_COLORS.keys())
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(8, 4.8))
    ax = plt.gca()

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = []
        route_node_indices = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(locations[node_index])
            route_node_indices.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
        route.append(locations[depot_index])
        route_node_indices.append(depot_index)
        route = np.array(route)
        color = colors[vehicle_id % len(colors)]

        plt.plot(route[:, 0], route[:, 1], marker='o', color=color, label=f'Veicolo {vehicle_id + 1}', linewidth=2, alpha=0.8)
        plt.plot(route[0, 0], route[0, 1], marker='D', markersize=10, color='black', markerfacecolor='yellow')

        for i in range(len(route)):
            x, y = route[i]
            node_index = route_node_indices[i]
            ax.annotate(str(node_index), (x, y), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title('Percorsi dei veicoli', fontsize=14)
    legend = ax.legend(loc='best', fontsize=10, framealpha=1, facecolor='white', edgecolor='black', shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig

# Colonna sinistra - Dati di input
with col1:
    st.markdown("<div class='column-title'>Dati di input</div>", unsafe_allow_html=True)
    with st.expander("Parametri"):  # Parametri in una sezione espandibile

        uploaded_file = st.file_uploader("Carica un'istanza  in formato CVRPlib", type=["vrp"])
        instance = None
        instance_name = ""
        # per disabilitare tutti i parametri di input e il button esegui, finchè non ho caricato un file
        is_disabled = uploaded_file is None
        if uploaded_file is not None:
            # Creiamo un file temporaneo per salvare il contenuto dell'istanza
            with tempfile.NamedTemporaryFile(delete=False, suffix=".vrp") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Lettura dell'istanza dal file temporaneo
            instance = vrplib.read_instance(temp_file_path)
            if 'edge_weight' in instance:
                instance.pop('edge_weight')
            instance_name = os.path.splitext(uploaded_file.name)[0]

        # Se l'istanza è caricata, calcoliamo il numero minimo di veicoli
        if instance is not None:
            total_demand = sum(instance['demand'][1:])
            default_num_vehicles = int(np.ceil(total_demand / instance['capacity']))
        else:
            default_num_vehicles = 1  # Valore di default se l'istanza non è caricata

        # Selezione del numero di veicoli
        num_vehicles_options = list(range(default_num_vehicles, default_num_vehicles + 10))
        num_vehicles = st.selectbox("Numero di veicoli", options=num_vehicles_options, index=0, disabled=is_disabled)

        # Selezione dell'euristica
        heuristic_list = [
            'AUTOMATIC', 'PATH_CHEAPEST_ARC', 'PATH_MOST_CONSTRAINED_ARC',
            'EVALUATOR_STRATEGY', 'SAVINGS', 'SWEEP', 'CHRISTOFIDES',
            'ALL_UNPERFORMED', 'BEST_INSERTION', 'PARALLEL_CHEAPEST_INSERTION',
            'LOCAL_CHEAPEST_INSERTION', 'GLOBAL_CHEAPEST_ARC', 'LOCAL_CHEAPEST_ARC',
            'FIRST_UNBOUND_MIN_VALUE'
        ]
        heuristic = st.selectbox("Seleziona l'euristica", options=heuristic_list, index=heuristic_list.index('PARALLEL_CHEAPEST_INSERTION'), disabled=is_disabled)

        # Scelta della metaeuristica
        use_metaheuristic = st.radio("Usare una metaeuristica?", options=['Sì', 'No'], horizontal=True, disabled=is_disabled)
        
        if use_metaheuristic == 'Sì':
            metaheuristic_list = [
                'AUTOMATIC', 'GREEDY_DESCENT', 'GUIDED_LOCAL_SEARCH',
                'SIMULATED_ANNEALING', 'TABU_SEARCH', 'GENERIC_TABU_SEARCH'
            ]
            metaheuristic = st.selectbox("Seleziona la metaeuristica", options=metaheuristic_list, index=metaheuristic_list.index('GUIDED_LOCAL_SEARCH'), disabled=is_disabled)
        else:
            metaheuristic = 'GREEDY_DESCENT'
            if not is_disabled:
                st.write("Metaeuristica selezionata: Nessuna metaeuristica")

        # Impostazione del time limit
        time_limit = st.number_input("Time limit (secondi)", min_value=1, value=50, disabled=is_disabled)

        # Bottone per eseguire l'algoritmo
        if st.button("Esegui", disabled=is_disabled):
            if instance is None:
                st.error("Per favore, carica un'istanza VRP prima di eseguire l'algoritmo.")
            else:
                with st.spinner('Risoluzione in corso...'):
                    # Creazione del data model
                    data = create_data_model(instance, num_vehicles)

                    # Creazione del manager e del modello di routing
                    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data['depot'])
                    routing = pywrapcp.RoutingModel(manager)

                    # Callback per le distanze
                    def distance_callback(from_index, to_index):
                        from_node = manager.IndexToNode(from_index)
                        to_node = manager.IndexToNode(to_index)
                        return data["distance_matrix"][from_node][to_node]

                    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
                    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

                    # Aggiunta dei vincoli di capacità
                    def demand_callback(from_index):
                        from_node = manager.IndexToNode(from_index)
                        return int(data["demands"][from_node])

                    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
                    routing.AddDimensionWithVehicleCapacity(
                        demand_callback_index,
                        0,
                        [int(capacity) for capacity in data["vehicle_capacities"]],
                        True,
                        "Capacity",
                    )
                    capacity_dimension = routing.GetDimensionOrDie("Capacity")

                    # Impostazione dei parametri di ricerca
                    search_parameters = get_search_parameters(heuristic, use_metaheuristic == 'Sì', metaheuristic, time_limit)

                    # Risoluzione del problema
                    start_time = time.time()
                    solution = routing.SolveWithParameters(search_parameters)
                    execution_time = time.time() - start_time

                    if solution:
                        # Ottenimento dei dati della soluzione
                        solution_data = get_solution_data(
                            data, manager, routing, solution, data['scale_factor'],
                            instance_name, heuristic, metaheuristic, execution_time
                        )
                        json_str = json.dumps(solution_data, indent=4, default=default)
                        solution_filename = f"solution_{instance_name}.json"

                        # Store variables in session state
                        st.session_state.solution_data = solution_data
                        st.session_state.json_str = json_str
                        st.session_state.solution_filename = solution_filename
                        st.session_state.solution = solution
                        st.session_state.data = data
                        st.session_state.manager = manager
                        st.session_state.routing = routing
                    else:
                        st.error("Nessuna soluzione trovata!")
                        st.session_state.solution_data = None
                        st.session_state.solution = None
                        #solution = None  # Assicurati che 'solution' non sia definito in caso di fallimento

# Colonna centrale - Grafici
with col2:
    if uploaded_file is not None:
        st.markdown("<div class='column-title'>Grafico dell'istanza senza soluzione</div>", unsafe_allow_html=True)
        fig1 = plot_instance(instance)
        st.pyplot(fig1)
    else:
        st.markdown("<div class='column-title'>Grafico dell'istanza senza soluzione</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='placeholder'>Carica un'istanza per visualizzare il grafico dell'istanza.</div>",
            unsafe_allow_html=True,
        )
    
    st.markdown("---")  # Linea di separazione tra i due grafici
    
    if st.session_state.solution is not None:
        st.markdown("<div class='column-title'>Grafico della soluzione</div>", unsafe_allow_html=True)
        fig2 = plot_solution(st.session_state.data, st.session_state.manager, st.session_state.routing, st.session_state.solution)
        st.pyplot(fig2)
    else:
        st.markdown("<div class='column-title'>Grafico della soluzione</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='placeholder'>Esegui l'algoritmo per visualizzare il grafico della soluzione.</div>",
            unsafe_allow_html=True,
        )

# Colonna destra - Dati di output
with col3:
    st.markdown("<div class='column-title'>Dati di output</div>", unsafe_allow_html=True)
    if st.session_state.solution_data is not None:
        # Your updated code for displaying output data
        # (Include the improved formatting as previously suggested)
        # Remember to use st.session_state.solution_data instead of solution_data

        # Visualizza le metriche principali in colonne
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric(label="Distanza totale", value=f"{st.session_state.solution_data['total_distance']:.2f} units")
        with col_metric2:
            st.metric(label="Obiettivo", value=f"{st.session_state.solution_data['objective']:.2f} units")

        # Seconda riga: Carico totale e Obiettivo
        col_metric3, col_metric4 = st.columns(2)
        with col_metric3:
            st.metric(label="Carico totale", value=f"{st.session_state.solution_data['total_load']} units")
        with col_metric4:
            st.metric(label="Tempo di esecuzione", value=f"{st.session_state.solution_data['execution_time']:.2f} sec")

        # Terza riga: Euristica (occupando entrambe le colonne)
        st.markdown("**Euristica utilizzata:**")
        st.write(f"{st.session_state.solution_data['heuristic']}")

        # Quarta riga: Metaeuristica (occupando entrambe le colonne)
        st.markdown("**Metaeuristica utilizzata:**")
        st.write(f"{st.session_state.solution_data['metaheuristic']}")
    
        st.markdown("---")
    
        # Visualizza i dettagli delle rotte
        st.subheader("Dettagli delle rotte")
        for vehicle_id in st.session_state.solution_data['routes']:
            route_info = st.session_state.solution_data['routes'][vehicle_id]
            with st.expander(f"Veicolo {vehicle_id}"):
                st.write(f"**Distanza della rotta:** {route_info['distance']:.2f} units")
                st.write(f"**Carico della rotta:** {route_info['load']} units")
                # Mostra la rotta in una tabella
                df_route = pd.DataFrame({
                    #'Ordine': range(len(route_info['route'])),
                    'Nodo': route_info['route']
                })
                st.table(df_route)
    
        st.markdown("---")
    
        # Pulsante per scaricare il file JSON
        st.download_button(
            label="Scarica soluzione in JSON",
            data=st.session_state.json_str,
            file_name=st.session_state.solution_filename,
            mime="application/json"
        )
    else:
        st.markdown(
            "<div class='placeholder'>Esegui l'algoritmo per visualizzare i dati di output.</div>",
            unsafe_allow_html=True,
        )
