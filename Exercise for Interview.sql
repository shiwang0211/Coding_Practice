-- three table join
select a.id, b.id, c.id from
	TableA as a 
	inner join TableB as b on a.id = b.id
	inner join TableC as c on b.id = c.id
	
-- sub-query select
select a.id, b.id from a
	inner join (select * from c) as b
	on a.id = b.id
