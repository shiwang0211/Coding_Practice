# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 20:57:04 2017

@author: shiwa
"""
aalist = [1,4,5,7,8,9,12]
aitem = 11
# Binary Search
def bsearch(alist, item):
    first = 0
    last = len(alist) - 1
    while (first <= last):
        mid = (first + last) // 2
        if alist[mid] == item:
            print ('aaaaa')
            return
        elif alist[mid] < item:
            first = mid +1
        elif alist[mid] > item:
            last = mid -1 
    return False

bsearch(aalist, aitem)

alist = [1,2,3,4,5,9]
target = 11

# Find 2 sum
def temp(alist, target):
    Set = set()
    for item in alist:
        if (target - item) in Set:
            print (item)
        else:
            Set.add(item)

# Find 2 sum by sorting
def temp(alist, target):
    alist.sort()
    first = 0
    last = len(alist) - 1
    
    while first < last:
        print(1)
        if (alist[first] + alist[last]) == target:
            print (alist[first])
            return
        if (alist[first] + alist[last]) < target:
            first += 1
        else:
            last -=1
    
        
# Finbonacci 

cache = {}
cache[1] = 1
cache[2] = 2

def fin(n):
    if n in cache:
        print (cache[n])
        return cache[n]
    else:
        cache[n] = fin(n-1) + fin(n-2)
        print (cache[n])
        return cache[n]


fin(10)

# Find leader larger than all elements to its right
alist = [6,7,8,9,7,11,3,6]

def find_leader(alist):
    last = len(alist) - 1
    t_max = alist[last]
    while last >=0:
        if alist[last - 1] > t_max:
            t_max = alist[last - 1]
        last -=1    
    print(t_max)    

# Determine parenthesis
lookup = {'(':')', '[':']'}
stack = []
alist='()[]'

def temp(alist):
    alist = list(alist)
    while alist:
        s = alist.pop(0)
        if s in lookup:
            stack.append(s) # add to end
            print(stack)
        
        # there should always be elemnts in the stack    
        elif not stack:
            print ('9999')
        else:
            s_s = lookup[stack.pop()] # pop from end
            if s != s_s:
                print ('8888')
                return False
temp(alist)    
    
# Define Tree
class Node:
    def __init__(self, key):
        self.data = key
        self.left = None
        self.right = None

root = Node(4)
b = Node(3)
c = Node(5)
d = Node(2.5)
e = Node(3.5)

root.left = b
root.right = c
b.left = d
b.right = e

# Breath first search
def BFT(root):
    queue = []
    queue.append(root) # add to end
    while queue:
        print (queue[0].data)
        node = queue.pop(0) # pop from 0
        if (node.left):
            queue.append(node.left)
        if (node.right):
            queue.append(node.right)

BFT(root)

last = [-100000]

# Depth first search
def DFT(root):
    if root.left:
        DFT(root.left)
        
    print(root.data)
    if root.data < last[0]: # decide is binary not not 
        print('NOT BST')
        return False
    last[0] = root.data
    
    if root.right:
        DFT(root.right)
        
DFT(root)

# Define linked list
class Node:
    def __init__(self, key):
        self.data = key
        self.next = None
head = Node(1)
b = Node(2)
c = Node(2)
d = Node(2)
e = Node(5)

head.next = b
b.next = c
c.next = d
d.next = e

# Remove same elements when they are neighbors
# use two pointers (一前一后)
def DL(head):
    cur_node = head
    node = cur_node.next
    while(node):
        while(node.data == cur_node.data):
            node = node.next # 2nd move, 1st stay
            if node is None:
                break
        cur_node.next = node
        cur_node = node
    
    node = head
    while(node):
        print(node.data)
        node=node.next
    
DL(head)

# Sort dictionary
aa = {'e':1, 'd':2, 'f':1.5}
for key, item in sorted(aa.items(), key = lambda x: x[1]):
    print (key, str(item))

print(aa.items())
print(aa.keys()) 

for key in sorted(aa.keys()):
    print (key, aa[key])
    
    
a= [1,4,3]
a.sort()
print(a)
print(sorted(a, key = lambda x: -x))

#Longest sub-sequency not continous (Difficult)
s = [30, 40, 20, 70, 10]

L = len(s)
Seg_Len = [0] * L
Prev = [-1] * L

for last in range(1,L):
    for first in range(0,last):
        if (s[last] > s[first]) and (Seg_Len[first] + 1 >  Seg_Len[last]):
            Seg_Len[last] += 1
            Prev[last] = first

Max_Seg_Len = 0
Max_Seg_Index = 0
for index, num in enumerate(Seg_Len):
    if num > Max_Seg_Len:
        Max_Seg_Len = num
        Max_Seg_Index = index

curr_index = Max_Seg_Index
final_list = []
while(curr_index >= 0):
    final_list.append(s[curr_index])
    curr_index =  Prev[curr_index]   

print(final_list)



itemList = ['hi', 'hi', 'hello', 'bye']

counter = {}
maxItemCount = 0
for item in itemList:
    try:
        # Referencing this will cause a KeyError exception
        # if it doesn't already exist
        counter[item]
        # ... meaning if we get this far it didn't happen so
        # we'll increment
        counter[item] += 1
    except KeyError:
        # If we got a KeyError we need to create the
        # dictionary key
        counter[item] = 1

    # Keep overwriting maxItemCount with the latest number,
    # if it's higher than the existing itemCount
    if counter[item] > maxItemCount:
        maxItemCount = counter[item]
        mostPopularItem = item

print mostPopularItem

# More on sorting
a = [('Al', 2),('Bill', 1),('Carol', 2), ('Abel', 3), ('Zeke', 2), ('Chris', 1)]  
b = sorted(sorted(a, key = lambda x : x[0]), key = lambda x : x[1], reverse = True)  
print(b)  
#[('Abel', 3), ('Al', 2), ('Carol', 2), ('Zeke', 2), ('Bill', 1), ('Chris', 1)]

alist = [(1,2,3,'a'),(1,2,3,'b'),(3,3,1,'c'),(2,9,3,'d'), (1,2,11,'e')]
sorted(alist, key = lambda x : (x[0], x[1], x[2]))
alist = ['1.2.3','1.2.4','2.9.2','1.1.9']
sorted(alist, key = lambda x: list(map(int, x.split('.'))))
