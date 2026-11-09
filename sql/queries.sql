-- 1. Monthly revenue and completed order count
SELECT strftime('%Y-%m',order_ts) AS month,ROUND(SUM(order_amount),2) AS revenue,COUNT(*) AS completed_orders FROM orders WHERE status='completed' GROUP BY month ORDER BY month;
-- 2. Top 10 customers by completed spend
SELECT c.customer_id,c.city,ROUND(SUM(o.order_amount),2) AS total_spend,COUNT(*) AS completed_orders FROM customers c JOIN orders o USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city ORDER BY total_spend DESC LIMIT 10;
-- 3. Revenue by product category
SELECT p.category,ROUND(SUM(o.order_amount),2) AS revenue,SUM(o.quantity) AS units FROM orders o JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY p.category ORDER BY revenue DESC;
-- 4. Delivery SLA by city
SELECT c.city,COUNT(*) AS measured_orders,SUM(CASE WHEN o.delivery_minutes>45 THEN 1 ELSE 0 END) AS late_orders,ROUND(100.0*SUM(CASE WHEN o.delivery_minutes>45 THEN 1 ELSE 0 END)/COUNT(*),2) AS late_pct FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' AND o.delivery_minutes IS NOT NULL GROUP BY c.city ORDER BY late_pct DESC;
-- 5. Ticket volume by category
SELECT true_category,COUNT(*) AS ticket_count FROM tickets WHERE true_category!='none' GROUP BY true_category ORDER BY ticket_count DESC;
-- 6. Customers with no completed orders
SELECT c.customer_id,c.city FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;
-- 7. Repeat purchase customers (at least three completed orders)
SELECT customer_id,COUNT(*) AS completed_orders,ROUND(SUM(order_amount),2) AS spend FROM orders WHERE status='completed' GROUP BY customer_id HAVING COUNT(*)>=3 ORDER BY completed_orders DESC,spend DESC;
