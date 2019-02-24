#!/usr/bin/env python
# -*-coding:utf-8-*- 
""" 
/* 
* @Author: zuofanxiu
* @Date: 12/6/18 2:37 AM 
* @file:leetcode.py
* @Software: PyCharm
*/
 """
import sys


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class MinStack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min = None


    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stack.append(x)
        if self.min == None or self.min > x:
            self.min = x


    def pop(self):
        """
        :rtype: void
        """
        popItem = self.stack.pop()
        if len(self.stack) == 0:
            self.min = None
        if popItem ==self.min:
            self.min = self.stack[0]
            for item in self.stack:
                if item < self.min:
                    self.min = item

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]


    def getMin(self):
        """
        :rtype: int
        """
        return self.min


class Solution():
    # 两数之和之暴力法
    def twoSum1(self, nums, target):
        """
        :type nums: list[int]
        :type target: int
        :rtype: list[int]
        """
        length = len(nums)
        for fir_index in range(length - 1):
            for sec_index in range(fir_index + 1, length):
                if nums[fir_index] + nums[sec_index] == target:
                    return [fir_index, sec_index]
        return None

    # 两数之和之数组转哈希表法
    def twoSum2(self, nums, target):
        """
        :type nums: list[int]
        :type target: int
        :rtype: list[int]
        """
        length = len(nums)
        data_dic = {}
        for fir_index in range(length):
            data_dic[nums[fir_index]] = fir_index
            sec_data = target - nums[fir_index]
            print("sec_data = ", sec_data)
            if sec_data in data_dic.keys() and sec_data != nums[fir_index]:
                sec_index = data_dic.get(sec_data)
                return [fir_index, sec_index]
        return None

    # 两数相加
    def addTwoNumber(self, l1, l2):
         """
         :type l1: list[int]
         :type l2: list[int]
         :rtype:
         """

    # 整数反转
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if -10 < x < 10:
            return x
        str_x = str(x)
        if str_x[0] != "-":
            str_x = str_x[::-1]
            x = int(str_x)
        else:
            str_x = str_x[1:][::-1]
            x = int(str_x)
            x = -x
        return x if -2147483648 < x < 2147483647 else 0

    # 回文数
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        x = str(x)
        if x == x[::-1]:
            return True
        else:
            return False

    # 罗马数字转整数
    def romanToInt(self, s):
        """
        :type s: str:
        :rtype: int
        """
        dic = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        res = 0
        for i in range(len(s)):
            print('i = ', i)
            if i < (len(s) - 1) and dic[s[i]] < dic[s[i+1]]:
                res -= dic[s[i]]
            else:
                res += dic[s[i]]
        return res

    # 最长公共前缀
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0 or strs is None:
            return ""
        else:
            i = 1
            prefix = strs[0]
            while i < len(strs):
                while not strs[i].startswith(prefix):
                    prefix = prefix[:len(prefix) - 1]
                    if prefix == "":
                        return ""
                i += 1
            return prefix

    # 有效的括号
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if s == "":
            return True
        else:
            dic_chars = {')': '(', '}': '{', ']': '['}
            pop_chars = {')', '}', ']'}
            n_stack = []
            for i in range(len(s)):
                if s[i] in pop_chars:
                    if len(n_stack) > 0 and n_stack[-1] == dic_chars[s[i]]:
                        n_stack.pop()
                    else:
                        return False
                else:
                    n_stack.append(s[i])
            if len(n_stack) == 0:
                return True
            else:
                return False

    # 合并两个有序链表
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 == None and l2 == None:
            return None
        if l1 == None:
            return l2
        if l2 == None:
            return l1
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

    # 删除排序数组中的重复项
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0 or len(nums) == 1:
            pass
        else:
            pre = nums[0]
            i = 1
            while i < len(nums):
                if nums[i] != pre:
                    pre = nums[i]
                    i += 1
                else:
                    nums.pop(i)
        return len(nums)

    # 移除元素
    def removeElement(self, nums, val):
        """
        :type nums: List(int)
        :type val: int
        :rtype: int
        """
        if len(nums) == 0:
            pass
        else:
            i = 0
            while i < len(nums):
                if nums[i] != val:
                    i += 1
                else:
                    nums.pop(i)
        return len(nums)

    # 实现strStr()
    def strStr(self, haystack, needle):
        return haystack.find(needle)

    # 接雨水
    def trap(self, height):
        """
        :type height: List[int]
        :rtype:  int
        """
        movepeak = 0
        rainArea = 0
        maxIndex = 0
        for i in range(1, len(height)):
            if height[maxIndex] < height[i]:
                maxIndex = i

        for j in range(maxIndex):
            if movepeak < height[j]:
                movepeak = height[j]
            else:
                rainArea += movepeak - height[j]

        movepeak = 0
        for k in range(len(height) - 1, maxIndex, -1):
            if movepeak < height[k]:
                movepeak = height[k]
            else:
                rainArea += movepeak - height[k]
        return rainArea

    # 搜索插入位置
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        for i in range(len(nums)):
            if target <= nums[i]:
                return i
        return len(nums)

    # 报数
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        pass

    # 最大子序列和
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        sum = nums[0]
        thisSum = 0
        for num in nums:
            if thisSum > 0:
                thisSum += num
            else:
                thisSum = num
            sum = max(thisSum, sum)
        return sum

    # 最后一个单词的长度
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype : int
        """
        s_list = s.strip().split(" ")
        if s_list[-1]:
            return len(s_list[-1])
        else:
            return 0

    # 加一
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        if len(digits) == 0:
            digits = [1]
        elif digits[-1] < 9:
            digits[-1] += 1
        else:
            digits = self.plusOne(digits[:-1])
            digits.extend([0])
        return digits

    #二进制求和
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        # 进位标志
        carry = 0
        c = []
        # c存储长数组，b存储短数组
        if len(a) >= len(b):
            c = list(a)
        else:
            c = list(b)
            b = list(a)
        min_len = len(b)
        max_len = len(c)
        for i in range(min_len):
                if int(c[max_len - 1-i])+int(b[min_len-1-i])+carry >= 2:
                    c[max_len - 1-i] = str(int(c[max_len - 1-i])+int(b[min_len-1-i])+carry - 2)
                    carry = 1
                elif int(c[max_len - 1-i])+int(b[min_len-1-i])+carry == 1:
                    c[max_len - 1-i] = '1'
                    carry = 0
                elif int(c[max_len - 1-i])+int(b[min_len-1-i])+carry == 0:
                    c[max_len - 1-i] = '0'
                    carry = 0

        for j in range(max_len-min_len-1, -1, -1):
            if int(c[j])+carry < 2:
                c[j] = str(int(c[j])+carry)
                carry = 0
            else:
                c[j] = '0'
                carry = 1

        if carry == 1:
            c = ['1'] + c
        res = ''.join(c)
        return res

    # 字符串相乘
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        num1 = list(num1[::-1])
        num2 = list(num2[::-1])
        arr = [0 for h in range(len(num1)+len(num2))]
        for i in range(len(num1)):
            for j in range(len(num2)):
                arr[i+j] += int(num1[i])*int(num2[j])
        res = []
        carry = 0
        for k in range(len(arr)):
            digit = (carry+arr[k]) % 10
            carry = (carry+arr[k])//10
            res.insert(0, str(digit))
        while res[0]=='0' and len(res)>1:
            del res[0]
        return ''.join(res)

    #x的平方根
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x <= 1:
            return x
        left = 0
        right = x
        while left < right:
            mid = (left + right) // 2
            if x // mid >= mid:
                left = mid + 1
            else:
                right = mid
        return right - 1

    # 爬楼梯
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 1 or n == 2:
            return n
        else:
            res = [1, 2]
            for i in range(3,n+1):
                temp = res[1]
                res[1] = res[0]+res[1]
                res[0] = temp
            return res[1]

    # 删除排序链表中的重复元素
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None or head.next == None:
            return head
        p = head
        while p.next:
            if p.val != p.next.val:
                p = p.next
            else:
                if p.next.next is None:
                    p.next = None
                    return head
                p.next = p.next.next
        return head

    # 合并两个有序数组
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void do not return anything, modify nums1 in-place instead
        """
        k = n+m-1
        while n > 0 and m > 0:
            if nums1[m-1] < nums2[n-1]:
                nums1[k] = nums2[n-1]
                n -= 1
            else:
                nums1[k] = nums1[m-1]
                m -= 1
            k -= 1
        while n > 0:
            nums1[k] = nums2[n-1]
            n -= 1
            k -= 1
        while m > 0:
            nums1[k] = nums1[m-1]
            m -= 1
            k -= 1
        return nums1

    # 相同的树
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p == None and q == None:
            return True
        if p == None or q == None:
            return False
        if p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False

    # 对称二叉树
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def isSame(p, q):
            if p == None and q == None:
                return True
            if p == None or q == None:
                return False
            if p.val == q.val:
                return isSame(p.left, q.right) and isSame(p.right, q.left)
            if p.val != q.val:
                return False

        if root == None:
            return True
        else:
            return isSame(root.left, root.right)

    # 二叉树的最大深度
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        else:
            return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    # 二叉树的层次遍历
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype : List[List[int]]
        """
        res = []
        que = [root]
        if root == None:
            return res
        while que:
            tempList = []
            for i in range(len(que)):
                node = que.pop(0)
                tempList.append(node.val)
                if node.left:
                    que.append(node.left)
                if node.right:
                    que.append(node.right)
            res.append(tempList)
        return res[::-1]

    # 将有序数组转换为二叉搜索树
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None
        mid = len(nums)//2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root

    # 平衡二叉树
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def height(node):
            if not node:
                return 0
            else:
                return 1 + max(height(node.left), height(node.right))
        if root is None:
            return True
        if abs(height(root.left) - height(root.right)) > 1:
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)

    # 二叉树的最小深度
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        if root.left == None and root.right == None:
            return 1
        if root.left == None:
            return 1+self.minDepth(root.right)
        if root.right == None:
            return 1+self.minDepth(root.left)
        else:
            return 1 + min(self.minDepth(root.left), self.minDepth(root.right))

    # 路径总和
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum:  int
        :rtype: bool
        """
        if root == None:
            return False
        if root.left == None and root.right == None:
            if root.val == sum:
                return True
            else:
                return False
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)

    # 杨辉三角
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype : List[List[int]]
        """
        res = []
        for i in range(numRows):
            now = [1]*(i+1)
            if i >= 2:
                for n in range(1, i):
                    now[n] = pre[n-1]+pre[n]
            pre = now
            res.append(now)
        return res

    # 杨辉三角II
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        res = []
        for i in range(rowIndex+1):
            res = [1]*(i+1)
            if i >= 2:
                for n in range(1, i):
                    res[n] = pre[n-1]+pre[n]
            pre = res
        return res

    # 买卖股票的最佳时机
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit = 0
        if len(prices) >= 2:
            buy = prices[0]
            for i in range(len(prices)):
                if prices[i] < buy:
                    buy = prices[i]
                else:
                    if prices[i]-buy > profit:
                        profit = prices[i] - buy
        return profit

    # 买卖股票的最佳时机
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit = 0
        for i in range(1, len(prices)):
            tmp = prices[i] - prices[i-1]
            if tmp > 0:
                profit += tmp
        return profit

    # 验证回文串
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = list(filter(str.isalnum, s.lower()))
        return True if s == s[::-1] else False

    # 只出现一次的数字
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a = 0
        for num in nums:
            a = a ^ num
            print(a)
        return a

    # 环形链表
    def hasCycle(self, head):
        """
        :type head: TreeNode
        :rtype: bool
        """
        fast = head
        slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
        return False

    # 相交链表
    def getIntersectionNode(self, headA, headB):
        """
        :type headA: ListNode
        :type headB: ListNode
        :rtype: ListNode
        """
        pA = headA
        pB = headB
        while pA != pB:
            pA = headB if pA == None else pA.next
            pB = headA if pB == None else pB.next
        return pA

    # 两数之和II - 输入有序数组
    def twoSum(self, numbers, target):
        """
        :type numbers: list[int]
        :type target:  int
        :rtype: list[int]
        """
        if len(numbers) <= 1:
            return None
        index1 = 0
        index2 = len(numbers)-1
        while index1 < index2:
            if numbers[index1] + numbers[index2] == target:
                return [index1+1, index2+1]
            if numbers[index1] + numbers[index2] < target:
                index1 += 1
            if numbers[index1] + numbers[index2] > target:
                index2 -= 1
        return None

    # Excel表列名称
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        result = ""
        while n != 0:
            result = chr((n-1) % 26 + 65) + result
            n = (n-1)//26
        return result

    # 求众数
    def majorityElement(self, nums):
        """
        :type nums: list[int]
        :rtype: int
        """
        count = 1
        maj = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == maj:
                count += 1
            else:
                count -= 1
                if count == 0:
                    maj = nums[i+1]
        return maj

    # Excel表列序号
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = 0
        base = 1
        for elem in s[::-1]:
            res += (ord(elem)-64)*base
            base *= 26
        return res

    # 阶乘后的零
    def trailingZeroes(self, n):
        """
        :type n: int
        :type: int
        """
        res = 0
        while n >= 5:
            n = n//5
            res += n
        return res

    # 旋转数组
    def rotate(self, nums, k):
        l = len(nums)
        nums[:l-k] = reversed(nums[:l-k])
        nums[l-k:] = reversed(nums[l-k:])
        nums[:] = reversed(nums)

    # 颠倒二进制数
    def reverseBits(self, n):
        """
        :type n: int
        :rtype: int
        """
        z = bin(n)[2:]
        z = "0"*(32-len(z))+z
        z = z[::-1]
        return int(z, 2)

    # 位１的个数
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = bin(n)[2:]
        count = 0
        for i in res:
            if i == "1":
                count += 1
        return count

    # 两数相加
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        p = ListNode((l1.val + l2.val) % 10)
        carry = (l1.val + l2.val) // 10
        head = p
        l1 = l1.next
        l2 = l2.next
        while l1 and l2:
            node = ListNode((l1.val + l2.val + carry) % 10)
            p.next = node
            p = p.next
            carry = (l1.val + l2.val + carry) // 10
            l1 = l1.next
            l2 = l2.next
        while l1:
            node = ListNode((l1.val + carry)%10)
            carry = (l1.val + carry)//10
            l1 = l1.next
            p.next = node
            p = p.next
        while l2:
            node = ListNode((l2.val + carry)%10)
            carry = (l2.val + carry)//10
            l2 = l2.next
            p.next = node
            p = p.next
        if carry == 1:
            p.next = ListNode(1)
        return head

    # 无重复字符的最长子串
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        maxNum = 0
        preNum = 0
        subStr = ""
        j = 0
        while s:
            while j < len(s)and s[j] not in subStr:
                subStr += s[j]
                j += 1
                preNum += 1
            if preNum > maxNum:
                maxNum = preNum
            if j >= len(s) - 1:
                return maxNum
            s = s[s.index(s[j])+1:]
            j = 0
            subStr = ""
            preNum = 0
        return maxNum

    # 寻找两个有序数组的中位数
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        len1 = len(nums1)
        len2 = len(nums2)
        if len1 == 0:
            if len2 % 2 == 0:
                return (nums2[len2//2-1]+nums2[len2//2])/2
            else:
                return nums2[len2//2]
        if len2 == 0:
            if len1 % 2 == 0:
                return (nums1[len1//2-1]+nums1[len1//2])/2
            else:
                return nums1[len1//2]
        nums = sorted(nums1+nums2)
        if (len2+len1) % 2 == 1:
            return nums[(len1+len2)//2]
        else:
            return (nums[(len1+len2)//2]+nums[(len1+len2)//2-1])/2





def main():
    solution = Solution()


if __name__ == '__main__':
    sys.exit(main())
