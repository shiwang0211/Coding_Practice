# -*- coding: utf-8 -*-

#
https://github.com/kamyu104/LeetCode
"""
Created on Fri Jun  2 18:29:43 2017

@author: shiwa
"""

 #136
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """ 
        #nums = list(map(int,input().split()))
        temp = 0
        for num in nums: 
            temp = temp ^ num
        return temp
         
#190
class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        result = 0
        for _ in range(32):
            result = result << 1
            result = result | (n & 1)
            n = n >> 1
        return result
    
#191
class Solution(object):
def hammingWeight(self, n):
    """
    :type n: int
    :rtype: int
    """
        temp = 0
        while n:
            n = n & (n-1)
            temp += 1
        return temp

#231   
class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """  
        if(n == 0):
            return (False)
        n = n & (n-1)
        return(True if n == 0 else False)
    
    
#342    
class Solution(object):
    def isPowerOfFour(self, num):
    """
    :type num: int
    :rtype: bool
    """
    while num and not (num & 0b11):
        num >>= 2
    return (num == 1)  
 
#389
class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        s = list(s)
        t = list(t)
        
        for i in s:
            t.remove(i)
        return t[0]

#461    
class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        bin(x^y).count('1')    
    
#477
class Solution(object):
    def totalHammingDistance(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        temp = 1
        ans = 0

        
        for _ in range(32):
            zeros = 0
            ones = 0
            for num in nums:
                if num & temp:
                    ones += 1
                else:
                    zeros += 1
            ans += ones * zeros
            temp <<= 1
        return(ans)


#6
    
class Solution(object):
def convert(self, s, numRows):
    step = 0 if numRows == 1 else -1
    rows = [''] * numRows
    row_idx = 0
    for c in s:
        rows[row_idx] += c
        if(row_idx == 0 or row_idx == numRows - 1):
            step = -step
        row_idx += step
    return(''.join(rows))
    

#14
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ''
        
        temp=''
        for index in range(len(strs[0])):
            for l in range(len(strs)):
                if index >= len(strs[l]):
                    return temp
                if  strs[0][index] !=  strs[l][index]:
                    return temp
            temp += strs[0][index]
        return(temp)    
    
#28
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if not needle:
            return 0
            
        return self.KMP(haystack, needle)
    
    def KMP(self, text, pattern):
        prefix = self.getPrefix(pattern)
        j = -1
        for i in xrange(len(text)):
            while j > -1 and pattern[j + 1] != text[i]:
                j = prefix[j]
            if pattern[j + 1] == text[i]:
                j += 1
            if j == len(pattern) - 1:
                return i - j
        return -1
    
    def getPrefix(self, pattern):
        prefix = [-1] * len(pattern)
        j = -1
        for i in xrange(1, len(pattern)):
            while j > -1 and pattern[j + 1] != pattern[i]:
                j = prefix[j]
            if pattern[j + 1] == pattern[i]:
                j += 1
            prefix[i] = j
        return prefix

    def strStr2(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        try:
            return haystack.index(needle)
        except:

 # 58          
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        temp = 0
        s = list(s)
        c = ''
        flag = 0
        while (True):
            try:
                c = s.pop()
                if c != ' ': 
                    temp +=1
                    flag = 1
                elif flag == 0:
                    continue
                else:
                    return temp  
            except:
                return temp
        
# 125
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not (s):
            return True
        
        i, j = 0, len(s) - 1
        
        while (i < j):
            while (i<j) and not s[i].isalnum():
                i = i +1
            while (i<j) and not s[j].isalnum():
                j = j -1
            if s[i].lower() != s[j].lower():
                return False
            i = i + 1
            j = j -1
        return True
                            



#26
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        last, i = 0, 1
        while (i<len(nums)):
            if nums[last] != nums[i]:
                nums[last + 1] = nums[i]
                last += 1
            i += 1

        return last+1

#27
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        last, i = 0, 0
        
        while(i<len(nums)):
            if nums[i] != val:
                nums[last] = nums[i]
                last +=1
            i +=1
        return last

#66
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry = 1
        temp = []
        for i in reversed(range(len(digits))):
            temp.append((digits[i]+carry) % 10)
            carry = (digits[i]+carry) // 10
            if i == 0 and carry == 1:
                temp.append(carry)
                
        return list(reversed(temp))   

#121
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        max_profit, min_price = 0, float('Inf')
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        return max_profit

# 21        
# iteratively
def mergeTwoLists1(self, l1, l2):
    
    dummy = cur = ListNode(0)
    while l1 and l2:
        if (l1.val < l2.val):
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    if not(l1):
        cur.next = l2
    if not (l2):
        cur.next = l1
    return dummy.next
    
    
# recursively    
def mergeTwoLists2(self, l1, l2):
    if not l1 or not l2:
        return l1 or l2
    if l1.val < l2.val:
        l1.next = self.mergeTwoLists(l1.next, l2)
        return l1
    else:
        l2.next = self.mergeTwoLists(l1, l2.next)
        return l2
        
#83
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        cur1 = cur2 = dummy.next = head
        
        if not head:
            return head
            
        if not head.next:
            return head
            
        while(cur1):
            temp = cur1.val
            while(cur2.val == temp):
                cur2 = cur2.next
                if not cur2:
                    break
            cur1.next = cur2
            cur1 = cur1.next
        
        return dummy.next
        
#20
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        lookup = {')':'(', ']':'[','}':'{'}
        s = list(s)
        stack = []
        while(s):
            temp = s.pop()
            if temp in lookup:
                stack.append(temp)
            elif not stack:
                return False
            elif lookup[stack.pop()] != temp:
                return False
          
        return True if len(stack) == 0 else False



#101

class Solution:
    # @param root, a tree node
    # @return a boolean
    def isSymmetric(self, root):
        if not root:
            return True
        stack = []
        stack.append(root.left)
        stack.append(root.right)
        
        while stack:
            x, y = stack.pop(), stack.pop()
            if not x and not y:
                continue
            if not x or not y or x.val != y.val:
                return False
            
            stack.append(x.right)
            stack.append(y.left)
            stack.append(x.left)
            stack.append(y.right)

        return True   

#226
class Solution(object):
    def invertTree(self, root):
        if root is not None:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        
        return root

#1            
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        Dict = {}
        for i,num in enumerate(nums):
            if (target - num) in Dict:
                return([Dict[target-num], i])
            else:
                Dict[num] = i
        return []
    
 #202   
 class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        temp_set = set()
        
        while n != 1:
             n = sum( [int(c) ** 2 for c in str(n)] )
             if n in temp_set:
                 return False
             temp_set.add(n)
        
        return True        

#409           
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        import collections
        
        num_odds = sum( map ( lambda x: x & 1, collections.Counter(s).values() ) )
        
        return len(s) - num_odds + (1 if num_odds > 0 else 0)
    
# recursive Fib
__fib_cache = {}
def fib_memo(n):
    if n in __fib_cache:
        return __fib_cache[n]
    
    else:
        __fib_cache[n] = n if n < 2 else fib_memo(n-2) + fib_memo(n-1)
        return __fib_cache[n]
