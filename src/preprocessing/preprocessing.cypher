// Rule 1
MATCH path = (start:Account)-[:TRANSFERRED_TO*]->(end:Account)
WHERE start <> end
WITH path, size(nodes(path)) AS pathLength
WITH AVG(pathLength) AS avgPathLength, COLLECT({path: path, length: pathLength}) AS paths
UNWIND paths AS pathInfo
WITH pathInfo, avgPathLength
WHERE pathInfo.length > avgPathLength
RETURN pathInfo.path AS path, pathInfo.length AS length, avgPathLength
ORDER BY length DESC


// Rule 4
MATCH path = (start:Account)-[:TRANSFERRED_TO*]->(end:Account)
WHERE start <> end
WITH path, nodes(path) AS pathNodes

// Unwind the path nodes to analyze each node individually
UNWIND pathNodes AS node

// Count outgoing and incoming TRANSFERRED_TO relationships using pattern comprehensions
WITH node,
     [n IN [(node)-[:TRANSFERRED_TO]->(x) | x] | n] AS outgoingRelationships,
     [n IN [(x)<-[:TRANSFERRED_TO]-(node) | x] | n] AS incomingRelationships

// Calculate the counts from the comprehensions
WITH node, SIZE(outgoingRelationships) AS outgoingCount, SIZE(incomingRelationships) AS incomingCount

// Filter nodes that have more than 1 incoming or outgoing relationship
WITH COLLECT(node) AS bifurcationPoints
WHERE SIZE(bifurcationPoints) >= 3  // At least 3 bifurcation points

// Return paths and bifurcation points
RETURN bifurcationPoints
ORDER BY SIZE(bifurcationPoints) DESC


// Rule 5
MATCH path = (start:Account)-[:TRANSFERRED_TO*]->(start)
WHERE length(path) > 2  // Exclude trivial cycles
RETURN path
ORDER BY length(path)
LIMIT 10
