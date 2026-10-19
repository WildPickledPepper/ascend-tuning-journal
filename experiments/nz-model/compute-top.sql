WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)
SELECT op_type, COUNT(*) AS calls, ROUND(SUM(duration_ns)/1000000.0,3) AS total_ms
FROM compute_view GROUP BY op_type ORDER BY total_ms DESC LIMIT 20;
