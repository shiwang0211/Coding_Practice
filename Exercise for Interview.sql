-- https://www.w3schools.com/sql/trysql.asp?filename=trysql_op_in
-- three table join
select a.id, b.id, c.id from
	TableA as a 
	inner join TableB as b on a.id = b.id
	inner join TableC as c on b.id = c.id
	
-- sub-query select
select a.id, b.id from a
	inner join (select * from c) as b
	on a.id = b.id

--- everything
SELECT  * FROM Customers as a
INNER JOIN (select * from Orders) as b
on a.CustomerID = b.CustomerID
where Country = 'USA'
group by City
having count(EmployeeID) > 2 

-- left join = full join + is not null check
SELECT DISTINCT * FROM Customers as a
LEFT JOIN (select * from Orders) as b
on a.CustomerID = b.CustomerID
where b.OrderID is not NULL

-- IN, between
SELECT DISTINCT * FROM Customers as a
INNER JOIN (select * from Orders) as b
on a.CustomerID = b.CustomerID
and a.Country in ('USA','Finland')
and b.EmployeeID BETWEEN 1 AND 5