# %% [bracket stack]

def solution(S):
    if len(S) == 0:
        return 1
    stack = []
    for s in S:
        if s in ['(', '{', '[']:
            stack.append(s)
        elif stack:
            if s == ')' and stack[-1] == '(':
                stack.pop()
            elif s == '}' and stack[-1] == '{':
                stack.pop()
            elif s == ']' and stack[-1] == '[':
                stack.pop()
            else:
                return 0 # 닫는 괄호가 매칭 안될 경우
        else:
            return 0 # ')', '}', ']'가 더 많은 경우
    if stack:
        return 0 # '(', '{', '['가 더 많은 경우
    else:
        return 1

# %% [Fish]

def solution(A, B):
    N = len(A)
    if N == 1:
        return 1
    else:
        stream_stack = []
        size_stack = []
        num_remain = N
        for i in range(N):
            if B[i] == 1: # downstream인 경우
                stream_stack.append(1)
                size_stack.append(A[i])
            elif B[i] == 0 and stream_stack: # upstream이면서 downstream인 fish 저장되어 있을 경우
                while stream_stack:
                    if A[i] < size_stack[-1]:
                        num_remain -= 1
                        break
                    else:
                        stream_stack.pop()
                        size_stack.pop()
                        num_remain -= 1
        return num_remain
    

# %% [괄호 한개]

def solution(S):
    if len(S) == 0:
        return 1
    stack = []
    for s in S:
        if s == '(':
            stack.append(1)
        elif stack:
            stack.pop()
        else:
            return 0
    if stack:
        return 0
    else:
        return 1
    
# %% ['AA', 'BB', 'CC']

def solution(S):
    stack = []
    for s in S:
        if stack and s == stack[-1]:
            stack.pop()
            continue
        stack.append(s)
    return ''.join(stack)

# %% [stonewall]

def solution(H):
    n = 0
    stack = []
    for h in H:
        while stack and stack[-1] > h:
            stack.pop()
        if not stack or stack[-1] < h:
            n += 1
            stack.append(h)
    return n

# %% [Palindrom]
def expend(s, left, right):
        while (0<=left) and (right<len(s)) and (s[left]==s[right]):
            left -= 1
            right += 1
        return s[left+1:right]


# %% 문제 1
def solution(S):
    dic = {}
    for s in S:
        if s in dic:
            return s
        else:
            dic[s] = 1

# %% 문제 2
def solution(blocks):
    N = len(blocks)
    if N == 2:
        return 2
    
    def max_len(x, left, right):
        while left-1 >= 0 and x[left-1] >= x[left]:
            left -= 1
        while right < len(x)-1 and x[right] <= x[right+1]:
            right += 1
        return right - left + 1
    
    n_list = []
    for i in range(N):
        n = max_len(blocks, i, i)
        n_list.append(n)
    return max(n_list)

# %% 문제 3
def solution(A, S):
    N = len(A)
    MAX = 1000000000
    total_sum = 0
    count = 0
    for i in range(N):
        total_sum += A[i]
        if total_sum == S * (i + 1):
            count += 1
        start = i - 1
        while start >= 0:
            total_sum -= A[start]
            if total_sum == S * (i - start):
                return

# %% 문제 3 다시 (성공)
def solution(A, S):
    A = [a - S for a in A]
    result = 0
    prefix_sum = 0
    MAX = 1000000000
    d = {0: 1}
    for n in A:
        prefix_sum += n
        if prefix_sum in d:
            result += d[prefix_sum]
            if result > MAX:
                return MAX
            d[prefix_sum] += 1
        else:
            d[prefix_sum] = 1
    return result

solution([1, -1] * 50000, 0)

# %% leetcode 560
class Solution:
    def subarraySum(self, nums, k):
        result = 0 # 가능한 경우의 수
        prefix_sum = 0 # 누적 합
        d = {0: 1} # 누적 합의 dictionary

        for n in nums:
            prefix_sum += n

            # (현재까지의 누적합 prefix_sum - 목표값 k)가 누적 합 dictionary에 있다면 
            # 목표값 k를 만들 수 있는 구간이 존재한다는 것
            # 가능한 경우의 수에 d[prefix_sum - k]만큼을 추가
            if prefix_sum - k in d:
                result += d[prefix_sum - k]

            # 딕셔너리의 값 채워 넣기
            if prefix_sum not in d:
                d[prefix_sum] = 1
            else:
                d[prefix_sum] += 1

        return result

a = Solution()
a.subarraySum(nums=[1, 1, 1], k=2)
# %% stonewall
def solution(H):
    n = 0
    stack = []
    for h in H:
        while stack and stack[-1] > h:
            stack.pop()
        if not stack or stack[-1] < h:
            n += 1
            stack.append(h)
    return n

print(solution([8, 8, 5, 7, 9, 8, 7, 4, 8]))
# %% 백준 11659 구간 합 구하기 4
import sys
input = sys.stdin.readline

N, M = map(int, input().split())
nums = list(map(int, input().split()))
cumsum = [0]
for i in range(N):
    cumsum.append(cumsum[-1] + nums[i])
for j in range(M):
    x, y = map(int, input().split())
    print(cumsum[y] - cumsum[x-1])

# %% Codility Prefix Sums - PassingCars
def solution(A):
    num_east = 0
    result = 0
    for a in A:
        if a == 0:
            num_east += 1
        else:
            result += num_east
    return result
solution([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])

# %% Codility Prefix Sums - CountDiv
def solution(A, B, K):
    b = B // K
    a = A // K if A % K != 0 else A // K - 1
    return b - a

solution(10, 20, 9)

# %% Leetcode 3. Longest Substring Without Repeating Characters
def solution(s):
    max_len = 0
    m = ''
    for x in s:
        if x not in m:
            m += x
        else:
            M = len(m)
            if max_len < M:
                max_len = M
            m += x
            idx = m.index(x)
            m = m[idx+1:]
        print(max_len)
        print(m, len(m))
    return max(max_len, len(m))

a = solution("baaaaa")
a
# %% 53. Maximum Subarray
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # Initialize our variables using the first element.
        current_subarray = max_subarray = nums[0]
        
        # Start with the 2nd element since we already used the first one.
        for num in nums[1:]:
            # 이전 current_subarray가 (-)이면 버림
            current_subarray = max(num, current_subarray + num)
            max_subarray = max(max_subarray, current_subarray)
            # print(current_subarray, max_subarray)
        
        return max_subarray

a = Solution()
a.maxSubArray([-2,1,-3,4,-1,2,1,-5,4])
# %% 55. Jump Game
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        limit = nums[0]
        for i, num in enumerate(nums):
            if i > limit:
                return False
            limit = max(limit, i + num)
            if limit >= len(nums) - 1:
                return True
        return True

# %%
def solution(A):
    arr = []
    for i, v in enumerate(A):
        arr.append((i-v, -1))
        arr.append((i+v, 1))

    arr.sort()
    intersection = 0
    intervals = 0
    print(arr)
    for i,v in enumerate(arr):
        if v[1] == 1 :
            intervals -= 1
        if v[1] == -1:
            intersection += intervals
            intervals += 1
    if intersection > 10000000:
        intersection = -1

    return intersection

solution([1, 5, 2, 1, 4, 0])
# %%
