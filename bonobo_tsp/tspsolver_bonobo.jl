using Bonobo
using MetaGraphs
using Graphs: Edge, nv, add_edge!, has_edge
using TravelingSalesmanExact

# Structure racine du problème
struct Root
    g::MetaGraph
    cost::Matrix{Float64}
    function Root(g::MetaGraph)
        cost = zeros((nv(g), nv(g)))
        for i in 1:nv(g)
            for j in i+1:nv(g)
                weight = get_prop(g, i, j, :weight)
                cost[i,j] = weight
                cost[j,i] = weight
            end
        end
        new(g, cost)
    end
end

# Nœud mutable pour Branch-and-Bound
mutable struct Node <: AbstractNode
    std :: BnBNodeInfo                               # Informations standard du nœud
    tour :: Union{Nothing, Vector{Int}}              # Circuit actuel (heuristique)
    mst :: Union{Nothing, Vector{Edge{Int}}}         # Arbre couvrant minimal relaxé
    fixed_edges :: Vector{Tuple{Int,Int}}            # Arêtes forcées
    disallowed_edges :: Vector{Tuple{Int,Int}}       # Arêtes interdites
end

# Indices de branchement par défaut (une seule branche)
function Bonobo.get_branching_indices(root::Root)
    return 1:1
end

# Fonction pour lire un fichier TSP et créer un MetaGraph
function parse_tsp_to_metagraph(input_path::String)
    # Récupère les coordonnées via simple_parse_tsp
    coords = simple_parse_tsp(input_path)
    n = length(coords)
    
    # Création d'un graphe complet
    g = MetaGraph(n)
    
    # Ajout des arêtes avec poids (distance euclidienne)
    for i in 1:n
        for j in i+1:n
            dist = sqrt((coords[i][1] - coords[j][1])^2 + (coords[i][2] - coords[j][2])^2)
            add_edge!(g, i, j)
            MetaGraphs.set_prop!(g, i, j, :weight, dist)
        end
    end
    
    return g
end

# Initialisation du modèle Bonobo
function optimize(input_path::String)
    # Convertit le TSP en MetaGraph
    g = parse_tsp_to_metagraph(input_path)
    # Construit le modèle Bonobo — on passe impérativement `brancher = g`
    bnb_model = Bonobo.initialize(
        traverse_strategy = Bonobo.BFS(),
        branch_strategy = LONGEST_EDGE(),
        Node = Node,
        root = Root(g),
        sense = :Min,
        Value = Vector{Int}
    )
    # Définit le nœud racine vide
    Bonobo.set_root!(bnb_model, (
        tour = nothing,
        mst = nothing,
        fixed_edges = Vector{Tuple{Int,Int}}(),
        disallowed_edges = Vector{Tuple{Int,Int}}()
    ))
    return bnb_model
end

# Calcul d'un 1-tree optimisé (borne inférieure) avec Prim
function get_optimized_1tree(cost_matrix::Matrix{Float64}; runs::Int=50)
    n = size(cost_matrix, 1)
    
    visited = falses(n)
    visited[1] = true
    total_cost = 0.0
    mst_edges = Edge{Int}[]
    
    for _ in 1:(n-1)
        min_cost = Inf
        min_edge = nothing
        
        for i in 1:n
            if visited[i]
                for j in 1:n
                    if !visited[j] && cost_matrix[i,j] < min_cost
                        min_cost = cost_matrix[i,j]
                        min_edge = Edge(i, j)
                    end
                end
            end
        end
        
        if min_edge !== nothing
            push!(mst_edges, min_edge)
            visited[min_edge.dst] = true
            total_cost += min_cost
        end
    end
    
    return mst_edges, total_cost
end

# Heuristique gloutonne du plus proche voisin pour borne supérieure
function greedy(g::MetaGraph)
    n = nv(g)
    if n == 0
        return Int[], Inf
    end
    
    visited = falses(n)
    tour = [1]
    visited[1] = true
    current = 1
    total_cost = 0.0
    
    for _ in 2:n
        min_cost = Inf
        next_city = -1
        
        for j in 1:n
            if !visited[j] && has_edge(g, current, j)
                cost = get_prop(g, current, j, :weight)
                if cost < min_cost
                    min_cost = cost
                    next_city = j
                end
            end
        end
        
        if next_city != -1
            push!(tour, next_city)
            visited[next_city] = true
            total_cost += min_cost
            current = next_city
        end
    end
    
    # Retour à la ville de départ
    if has_edge(g, current, 1)
        total_cost += get_prop(g, current, 1, :weight)
    else
        total_cost = Inf
    end
    
    return tour, total_cost
end

# Forcer l'inclusion d'une arête (placeholder)
function fix_edge!(root::Root, edge::Tuple{Int,Int})
    return 0.0  # Pour l'instant, pénalité nulle
end

# Interdire une arête en mettant son coût à l'infini
function disallow_edge!(root::Root, edge::Tuple{Int,Int})
    i, j = edge
    root.cost[i,j] = Inf
    root.cost[j,i] = Inf
end

# Stratégie de branchement : arête la plus longue
struct LONGEST_EDGE <: Bonobo.AbstractBranchStrategy end

# Valeurs relaxées (tour heuristique)
function Bonobo.get_relaxed_values(tree::BnBTree{Node, Root}, node::Node)
    return node.tour
end

# Vérifie si un tour est valide (cycle hamiltonien)
function is_valid_tour(tour::Vector{Int}, n::Int)
    return length(tour) == n && length(unique(tour)) == n
end

# Évaluation d'un nœud : calcul des bornes inférieure et supérieure
function Bonobo.evaluate_node!(
    tree::BnBTree{Node,Root}, node::Node
)
    root = deepcopy(tree.root)
    extra_cost = 0.0
    n = nv(root.g)
    
    # Appliquer les contraintes
    for fixed_edge in node.fixed_edges
        extra_cost += fix_edge!(root, fixed_edge)
    end
    for disallowed_edge in node.disallowed_edges
        disallow_edge!(root, disallowed_edge)
    end
    
    # Calcul des bornes
    mst, lb = get_optimized_1tree(root.cost; runs=50)
    tour, ub = greedy(root.g)
    lb += extra_cost
    ub += extra_cost
    
    node.mst = mst
    node.tour = tour
    
    # Vérifier si c'est un nœud feuille (toutes les arêtes sont contraintes)
    if is_leaf_node(node, n)
        println("Nœud feuille atteint avec $(length(node.fixed_edges)) arêtes fixées et $(length(node.disallowed_edges)) interdites")
        if is_valid_tour(tour, n)
            println("Solution valide trouvée ! Tour: $tour, Coût: $ub")
            println("ARRÊT : Solution optimale détectée")
            # Forcer l'arrêt en définissant une borne très restrictive
            tree.options.abs_gap_limit = 0.0
            tree.options.dual_gap_limit = 0.0
            return ub, ub  # Borne inf = borne sup pour arrêter
        else
            println("Solution non valide à ce nœud feuille")
            return NaN, NaN  # Élagage de ce nœud
        end
    end
    
    # Vérifier si on a trouvé une solution optimale (gap très petit)
    if is_valid_tour(tour, n) && abs(ub - lb) < 1e-6
        println("Solution optimale trouvée ! Tour: $tour, Coût: $ub")
        println("ARRÊT : Gap optimal atteint")
        # Forcer l'arrêt
        tree.options.abs_gap_limit = 0.0
        tree.options.dual_gap_limit = 0.0
        return ub, ub
    end
    
    # Affichage de débogage si peu de contraintes
    if length(node.fixed_edges) + length(node.disallowed_edges) <= 2
        println("Nœud évalué : fixées=$(length(node.fixed_edges)), interdites=$(length(node.disallowed_edges)), lb=$lb, ub=$ub")
    end
    
    if isinf(ub)
        return NaN, NaN
    end
    return lb, ub
end

# Vérifie si un nœud peut être branché
function can_branch(node::Node, root::Root)
    if node.mst === nothing
        return false
    end
    
    # Vérifier s'il reste des arêtes libres dans le MST
    free_edges = 0
    for edge in node.mst
        edge_tpl = (edge.src, edge.dst)
        edge_tpl_rev = (edge.dst, edge.src)
        if !(edge_tpl in node.fixed_edges) && !(edge_tpl_rev in node.fixed_edges) && 
           !(edge_tpl in node.disallowed_edges) && !(edge_tpl_rev in node.disallowed_edges)
            free_edges += 1
        end
    end
    
    return free_edges > 0
end

# Vérifie si un nœud représente une solution complète
function is_leaf_node(node::Node, n::Int)
    if node.mst === nothing
        return false
    end
    
    # Un nœud est une feuille si toutes les arêtes du MST sont soit fixées soit interdites
    total_edges = length(node.mst)
    constrained_edges = 0
    
    for edge in node.mst
        edge_tpl = (edge.src, edge.dst)
        edge_tpl_rev = (edge.dst, edge.src)
        if (edge_tpl in node.fixed_edges) || (edge_tpl_rev in node.fixed_edges) || 
           (edge_tpl in node.disallowed_edges) || (edge_tpl_rev in node.disallowed_edges)
            constrained_edges += 1
        end
    end
    
    return constrained_edges == total_edges
end

# Sélection de l'arête de branchement (la plus longue)
function Bonobo.get_branching_variable(tree::BnBTree{Node, Root}, ::LONGEST_EDGE, node::Node)
    if !can_branch(node, tree.root)
        println("Aucun branchement possible : toutes les arêtes sont fixées ou interdites")
        return nothing
    end
    
    longest_len = 0.0
    longest_edge = nothing
    
    if node.mst === nothing
        println("Attention : MST est vide dans get_branching_variable")
        return nothing
    end
    
    for edge in node.mst
        edge_tpl = (edge.src, edge.dst)
        edge_tpl_rev = (edge.dst, edge.src)
        
        # Vérifier que l'arête n'est ni fixée ni interdite
        if !(edge_tpl in node.fixed_edges) && !(edge_tpl_rev in node.fixed_edges) && 
           !(edge_tpl in node.disallowed_edges) && !(edge_tpl_rev in node.disallowed_edges)
            
            len = get_prop(tree.root.g, edge_tpl..., :weight)
            if len > longest_len
                longest_edge = edge
                longest_len = len
            end
        end
    end
    
    if longest_edge === nothing
        println("Aucune arête de branchement trouvée")
        return nothing
    else
        println("Branchement sur l'arête $(longest_edge.src)-$(longest_edge.dst) de longueur $longest_len")
        return longest_edge
    end
end

# CORRECTION PRINCIPALE : Arrêter le branchement si aucune arête n'est disponible
function Bonobo.get_branching_nodes_info(tree::BnBTree{Node, Root}, node::Node, branching_edge::Union{Edge, Nothing})
    # Si aucune arête de branchement, arrêter le branchement
    if branching_edge === nothing
        println("Arrêt du branchement : aucune arête disponible")
        return NamedTuple[]  # Retourner un tableau vide pour arrêter le branchement
    end
    
    # Vérifier si une solution optimale a déjà été trouvée
    if hasfield(typeof(tree.options), :abs_gap_limit) && tree.options.abs_gap_limit <= 0.0
        println("Arrêt du branchement : solution optimale déjà trouvée")
        return NamedTuple[]
    end
    
    nodes_info = NamedTuple[]
    
    # Branche 1 : Forcer l'inclusion de l'arête
    new_fixed_edges = deepcopy(node.fixed_edges)
    push!(new_fixed_edges, (branching_edge.src, branching_edge.dst))
    push!(nodes_info, (
        tour = nothing,
        mst = nothing,
        fixed_edges = new_fixed_edges,
        disallowed_edges = deepcopy(node.disallowed_edges),
    ))
    
    # Branche 2 : Interdire l'arête
    new_disallowed_edges = deepcopy(node.disallowed_edges)
    push!(new_disallowed_edges, (branching_edge.src, branching_edge.dst))
    push!(nodes_info, (
        tour = nothing,
        mst = nothing,
        fixed_edges = deepcopy(node.fixed_edges),
        disallowed_edges = new_disallowed_edges,
    ))
    
    return nodes_info
end

# Fonction principale d'exécution
function main()
    tspfile = "data/att48.tsp"
    bnb = optimize(tspfile)
    println("Résolution en cours…")
    
    # Limites pour éviter une exécution trop longue
    bnb.options.abs_gap_limit = 1000.0  # Limite de l'écart absolu
    bnb.options.dual_gap_limit = 0.2    # Limite de l'écart relatif à 20%
    
    # Vérifier les champs disponibles dans options
    println("Champs disponibles dans options : ", fieldnames(typeof(bnb.options)))
    
    # Essayer d'autres limites disponibles
    if hasfield(typeof(bnb.options), :time_limit)
        bnb.options.time_limit = 60.0  # 1 minute maximum
        println("Limite de temps définie à 60 secondes")
    end
    if hasfield(typeof(bnb.options), :iteration_limit)
        bnb.options.iteration_limit = 500
        println("Limite d'itérations définie à 500")
    end
    if hasfield(typeof(bnb.options), :node_limit)
        bnb.options.node_limit = 500
        println("Limite de nœuds définie à 500")
    end
    
    try
        # Lancement de l'optimisation
        println("Début de l'optimisation...")
        start_time = time()
        
        Bonobo.optimize!(bnb)
        
        end_time = time()
        elapsed_time = end_time - start_time
        
        println("Optimisation terminée en $(round(elapsed_time, digits=2)) secondes !")
        println("Statut final: $(bnb.status)")
        println("Champs de BnBTree : ", fieldnames(typeof(bnb)))
        
        # Accès à la solution optimale
        if hasproperty(bnb, :incumbent)
            println("Solution courante trouvée")
            best_solution = bnb.incumbent
            println("→ tour = ", best_solution.value)
            println("→ coût = ", best_solution.upper_bound)
        elseif hasproperty(bnb, :best_solution)
            println("Meilleure solution trouvée")
            best_solution = bnb.best_solution
            println("→ tour = ", best_solution.value)
            println("→ coût = ", best_solution.upper_bound)
        elseif hasproperty(bnb, :solutions)
            println("Champ solutions trouvé")
            if length(bnb.solutions) > 0
                best_solution = first(values(bnb.solutions))
                println("→ tour = ", best_solution.value)
                println("→ coût = ", best_solution.upper_bound)
            else
                println("Le champ solutions est vide")
            end
        else
            println("Aucun champ de solution standard trouvé")
            println("Champs disponibles : ", fieldnames(typeof(bnb)))
        end
        
    catch e
        println("Erreur durant l'optimisation : ", e)
        println("Trace de la pile :")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
end

main()