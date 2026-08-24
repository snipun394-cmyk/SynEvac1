def bfs_reachable(graph, start_ids, is_traversable, excluded_node_ids=frozenset()):

    # Extracted, verbatim, from scenario_validator.navigation_validation's
    # own private `_bfs_reachable()` (Scenario Campaign Feasibility
    # Preflight Phase 1) -- moved here, unchanged, because it has no
    # dependency on Scenario/candidate types beyond what its caller
    # passes in (a NavigationGraph, a set of start node ids, an
    # injectable traversability predicate, and an optional excluded-node
    # set). navigation_validation.py imports this instead of defining
    # its own copy; the Campaign Feasibility Preflight (campaign_
    # feasibility/) reuses the same function for the same reason,
    # rather than a second, independently-written reachability
    # algorithm. Behavior is byte-for-byte identical to the function
    # this replaced -- see navigation_validation.py's own module
    # docstring for why a candidate's resolved states, not
    # Edge.traversable, drive `is_traversable`.

    reachable = set()
    frontier = [node_id for node_id in start_ids if node_id not in excluded_node_ids]

    while frontier:

        current = frontier.pop()

        if current in reachable or current in excluded_node_ids:
            continue

        reachable.add(current)

        current_node = graph.find_node(current)

        if current_node is None:
            continue

        for neighbor_node, edge in graph.find_neighbors(current_node):

            if neighbor_node.id in excluded_node_ids or neighbor_node.id in reachable:
                continue

            if is_traversable(edge):
                frontier.append(neighbor_node.id)

    return reachable
