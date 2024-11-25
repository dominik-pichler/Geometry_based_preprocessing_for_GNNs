MATCH path = (start:Account)-[rel:TRANSFERRED_TO]->(end:Account)
WHERE start <> end
WITH avg(size(nodes(path))) AS averagePathLength

// Rule 1: Paths shorter than the average path length
MATCH path = (start:Account)-[rel:TRANSFERRED_TO]->(end:Account)
WHERE start <> end AND size(nodes(path)) < averagePathLength
RETURN start as from_id,
       end as to_id,
       [rel IN relationships(path) | properties(rel)] AS all_relationship_attributes
UNION

//Rule 4: Nodes with more than one outgoing or incoming relationship
MATCH path = (start:Account)-[r:TRANSFERRED_TO]->(end:Account)
WITH start AS n, end AS m, COUNT(r) AS branch_count, COUNT(start) AS incoming_count, path
WHERE branch_count > 1 OR incoming_count > 1
RETURN n AS from_id,
       m AS to_id,
       [rel IN relationships(path) | properties(rel)] AS all_relationship_attributes
UNION

//Rule 5: Non-trivial cycles (paths longer than 2)
MATCH path = (start:Account)-[:TRANSFERRED_TO*]->(end)
WHERE length(path) > 2 // Exclude trivial cycles
RETURN start as from_id,
       end as to_id,
       [rel IN relationships(path) | properties(rel)] AS all_relationship_attributes