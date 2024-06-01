package blind75

import java.util.*
import kotlin.collections.HashMap
import kotlin.math.max

//https://dsa.30dayscoding.com/


/**
 * Time Complexities
 *  1. UnorderedMap lookup - O(1)
 *      - HaspMap id unordered map
 *
 *  2. orderedMap lookup - O(log n), as it has to maintain the order or keys inserted
 *      - LinkedHashMap is unorderd map
 *      - mutableMapOf() gives us ordered map
 */


/**
 * ASCII Characters: 128 characters
 * A-Z: 65 to 90
 * a-z: 97 to 122
 *
 * https://www.techtarget.com/whatis/definition/ASCII-American-Standard-Code-for-Information-Interchange#:~:text=Characters%20in%20ASCII%20encoding%20include,use%20with%20teletype%20printing%20terminals.
 */


// Pattern : Frequency map

/**
 * Problem: https://leetcode.com/problems/contains-duplicate/
 *
 * Approach: Frequency Map to keep track for frequency while traversing aray
 *
 * Note: We can use set also and compare size but Map is more efficient as we can break out early if we found dup element
 */
fun containsDuplicate(nums: IntArray): Boolean {
    val map = mutableMapOf<Int, Int>()
    nums.forEach {
        if (map.containsKey(it)) return true
        map[it] = 1
    }
    return false
}



/**
 * Problem: https://leetcode.com/problems/valid-anagram/
 * Approach : Frequency map/array for characters
 *  Here given that s and t contains lowercase english chars only that why we have took array of size 26
 */
fun isAnagram(s: String, t: String): Boolean {
    if (s.length != t.length) return false

    val counts = IntArray(26) // Assuming ASCII characters, lowercase characters: 26 size


    for (char in s) {
        counts[char - 'a']++
    }

    for (char in t) {
        val index = char - 'a'
        if (counts[index] == 0) return false
        counts[index]--
    }

    return true
}


/**
 * Problem: https://leetcode.com/problems/two-sum/
 * Approach 1: Hashing
 *
 * The idea is that we should remember the numbers we have already seen, we will store visited numbers in hashmap with their index
 * and lookup for desired number in the constant time.
 *
 * This will add some space complexity
 *
 * Time complexity: O(n * logn/1): n --> iteration, ordered map lookup: O(logn)  and unordered map lookup: O(1)
 * Space complexity: O(n): size of hashmap
 *
 *
 * Note: If we have to return YES/NO like if the input array has the solution or not?  we can use 2 Pointer approach
 * 1.Sort array
 * 2.Place 2 pointer at start and end of array
 * 3. Check if target = start + end
 * 4. if target is less --> move end to left (moving end to left means we are reducing the number as array is sorted)
 * 5. if target is more --> move start to right  (moving start to rught means we are increasing the number as array is sorted)
 * 6. Do it untill start < end
 *
 * Reference: https://www.youtube.com/watch?v=UXDSeD9mN-k&ab_channel=takeUforward
 */
fun twoSum(nums: IntArray, target: Int): IntArray {

    // We will keep number we have already seen with their index
    val seenMap = hashMapOf<Int, Int>()
    val ans = IntArray(2)
    nums.forEachIndexed { index, i ->
        // partner: What we required to achieve target
        val partner = target - i
        if (seenMap.containsKey(partner)) {
            // Map has the partner --> we found ans
            ans[0] = seenMap[partner]!!
            ans[1] = index
            return ans
        } else {
            // We have not encounter partner yet in array
            seenMap[i] = index
        }
    }
    return ans
}


/**
 * Problem: https://leetcode.com/problems/group-anagrams/description/
 *
 * Approach: Hashing
 *
 * The Idea is that to traverse the array only once and we will identify anagram by comparing the sorted value of strings
 *
 *  Time complexity: O(n * k * logk): n --> iterating on items, k * logk --> sorting a string if length k
 * Space complexity: O(n * k): size of hashmap
 *
 */
fun groupAnagrams(strs: Array<String>): List<List<String>> {
    val map = HashMap<String, MutableList<String>>()
    strs.forEach {
        val key = it.toCharArray().sorted().toString()
        map.putIfAbsent(key, arrayListOf())
        map[key]?.add(it)
    }

    // Preparing ans from map
    return map.values.toList()
}


/**
 * Problem: https://leetcode.com/problems/top-k-frequent-elements/submissions/1234169661/
 *
 * Approach :
 * There are 2 parts of the problem
 *  1. Find Frequency --> Frequency Map
 *  2. Get top k elements based on frequency --> sort frequency map based on freqs
 *
 * the time complexity depends on the step 2. where we sort based on frequency
 * we can use normal sort O(n * logn), Heap sort O(n * logk), Bucket Sort O(n)
 *
 * Ref: https://www.youtube.com/watch?v=YPTqKIgVk-k&t=184s&ab_channel=NeetCode
 *
 */
fun topKFrequent_1(nums: IntArray, k: Int): IntArray {

    // Prepare frequency map O(n)
    val freqMap = hashMapOf<Int, Int>()
    for (num in nums) {
        freqMap[num] = freqMap.getOrDefault(num, 0) + 1
    }

    // Sort Map based on frequency O(n * logn)
    val sortedMap = freqMap.toList().sortedByDescending { it.second }.toMap()

    return sortedMap.keys.take(k).toIntArray()
}

// Here we are using bucket sort concept
fun topKFrequent_2(nums: IntArray, k: Int): IntArray {

    // Prepare frequency map O(n)
    val freqMap = hashMapOf<Int, Int>()
    for (num in nums) {
        freqMap[num] = freqMap.getOrDefault(num, 0) + 1
    }

    // Sort Map using bucket sort O(n)

    /**
     * Prepare buckets
     * Note: Input elements (nums) does not have any limitation on range (not bounded) and not sure about their uniform distribution
     * BUT BUT BUT
     * The frequencies are bounded(0 - n). n --> when all elements are same in nums
     * So what if we create a bucket for each frequency 0-n and
     */
    val n = nums.size
    val buckets = Array(n + 1){ hashSetOf<Int>()}

    // loop over frequency and populate buckets
    freqMap.forEach { item, freq ->
        buckets[freq].add(item)
    }

    // get data from buckets no. n to 1
    val ans = mutableListOf<Int>()
    for (i in n downTo 0) {
        val items = buckets[i]
        if (items.isNotEmpty()) {
            ans.addAll(items)
        }
    }

    return ans.take(k).toIntArray()
}


/**
 * Problem: https://leetcode.com/problems/product-of-array-except-self/description/
 *
 * Approach/Pattern : Prefix and postfix.
 * Here we store the calculation of elements previous to arr[i] while traversing in a prefix array (product for this problem)
 * Also create a post fix array with same concept and derive result form these arrays
 *
 *
 * Note: Here we are creating 3 diff arrays: prefix, postfix and ans. But we can avoid 1 array, by directly storing result in any one array
 */
fun productExceptSelf(nums: IntArray): IntArray {
    val n = nums.size
    // Create prefix product array
    val preArr = IntArray(n){1}
    for (i in 1 until n) {
        preArr[i] = nums[i - 1] * preArr[i - 1]
    }

    val postArr = IntArray(n){1}
    for (i in n-2 downTo 0) {
        postArr[i] = nums[i + 1] * postArr[i + 1]
    }

    val ans =  IntArray(n){
        preArr[it] * postArr[it]
    }

    return ans
}


/**
 * Problem: https://leetcode.com/problems/longest-consecutive-sequence/description/
 *
 * Approach/Pattern:
 * - Use HashSet for lookup: HashSet are implemented using HashMap so the HashSet.contains() take constant time i.e O(1)
 * - Find Start of sequence and check for next elements in loop: An element x is the start of the consecutive seq. if x - 1 is not present in arr.
 *    Ex: arr[3,7,2,5,8,4,6, 102, 101, 100, 99] - there are 2 sequences.
 *      1. 2,3,4,5,6,7,8 :  2 is start of seq. as 2-1 = 1 is not present
 *      2. 99, 100, 101, 102: 99 is start of seq. as 99-1 = 98 is not present
 *  If a element is start of a seq. we can look for next elements in a loop
 *
 *  Key here is that lookup in a Set is O(1)
 *
 * ref: https://www.youtube.com/watch?v=P6RZZMu_maU&t=460s&ab_channel=NeetCode
 */

fun longestConsecutive(nums: IntArray): Int {
    val lookupSet = nums.toHashSet()
    var longest = 0

    for (i in nums) {
        // Check if i is the start of the sequence
        if (lookupSet.contains(i - 1).not()) {  // Note: time complexity of "contains" function in HashSet is O(1)
            var length = 1
            var next = i + 1
            while (lookupSet.contains(next)) {
                length ++
                next ++
            }
            longest = maxOf(longest, length)
        }
    }

    return longest

}


/**
 * Problem = https://leetcode.com/problems/valid-palindrome/
 *
 * Approach: 2 pointer approach
 *
 * Note: We can further optimise this by ignoring the non alphanumeric char and converting to lowercase while comparision in the loop itself.
 * That will save us the extra time of filtering and extra space
 *
 */
fun isPalindrome(s: String): Boolean {
    // filter non alphanumeric characters
    val input = s.lowercase().filter { it.isLetterOrDigit() }.toCharArray()

    // iterate on array using 2 pointers
    var i = 0
    var j = input.size - 1

    while (i < j) {
        if (input[i] != input[j]) return false
        i++
        j--
    }

    return true
}


/**
 * Problem: https://leetcode.com/problems/3sum/submissions/1237390608/
 *
 * approach 1: brute force O(n*n*n)
 * 1. Iterate over array using i,j,k
 * 2. check for each triplet.
 * 3. sort items in triplet so that we can compare, to avoid duplicate (as array can contains dups)
 * 4. store triplet in set to avoid duplicate triplet
 *
 *
 * Approach 2: Keep one variable fix and perform 2 sum on rest
 *  1. We need sorted array
 *  2. Loop i on array and apply 2 pointer approach (j, k) on arr i+1 to n
 *  3. Imp: to avoid duplicate triplet, we will not only move j,k by 1 index, instead we will move them till next different number
 *  Time Complexity: O(nLogn) for sorting + O(n * n) n for i looping on whole arr, n for j and k covers rest of the arr = O(n*n)
 *  Space Complexity: O(3) for triplet, which is constant + O(triplet in ans) ~ O(1)
 *
 *
 * Ref: https://www.youtube.com/watch?v=DhFh8Kw7ymk&t=1915s&pp=ygUJMyBzdW0gRFNB
 */


fun threeSum(nums: IntArray): List<List<Int>> {
    val n = nums.size
    nums.sort()
    val ans = mutableListOf<List<Int>>()

    for (i in 0 until n - 2) {
        // Move i till we get a different number than previous
        if (i > 0 && nums[i] == nums[i-1]) continue
        var j = i + 1
        var k = n - 1

        while (j < k) {
            val sum = nums[i] + nums[j] + nums[k]
            if (sum == 0) {
                // found triplet
                ans.add(listOf(nums[i] , nums[j] , nums[k]))

                // incrementing j & k to next different value
                while ( j < k && nums[j] == nums[j + 1]) j ++
                while ( j  < k  && nums[k] == nums[k - 1]) k --
                j++
                k--

            } else if (sum < 0) {
                while ( j < k && nums[j] == nums[j + 1]) j ++
                j++

            } else {
                while ( j  < k  && nums[k] == nums[k - 1]) k --
                k--

            }
        }
    }

    return ans

}


/**
 * Problem: https://leetcode.com/problems/container-with-most-water/submissions/1237409533/
 *
 * Approach: 2 Pointer
 * Note: Why we are moving pointer of smaller height???
 * Ans: lets say i = 0, height[i] = 6, j = 5, height[j] = 8
 * area = min(8, 6) * (5 - 0)
 * Now
 * case 1: If I move j: if the next height is > 8 then also area will not increase as height[i] is smaller and it controls the height. if the next height is < 8
 * then also area will decrease according to new height. i.e in any case if we move a bigger height we will not get better solution.
 *
 * Case 2: If I move i: if the next height is > 6 then also area will increase as height[j] = 8 and we will get new height anyways > 6 only. if the next height is < 6
 * then also area will decrease according to new height. i.e in case one case we have chance of better solution

 * That's why we move smaller height
 *
 * Time complexity: O(n)
 * Space complexity: O(1)
 *
 *
 */
fun maxArea(height: IntArray): Int {
    var max = 0
    var i = 0;
    var j = height.size - 1

    while (i < j) {
        val current = (j - i)* minOf(height[i], height[j])
        max = maxOf(max, current)

        // IMP: move smaller height in hope of better solution
        if (height[i] <= height[j]) {
            i ++
        } else if(height[i] > height[j]) {
            j --
        }
    }

    return max
}

/**
 * Problem: https://leetcode.com/problems/sort-colors/submissions/1238258273/
 * Approach 1: Frequency counters
 *  1. Count number of 0,1 and 2 by looping through array
 *  2. rewrite array according to the count
 *
 *  Time Complexity: O(2n) : iterating 2 times
 *  Space Complexity: O(1)
 *
 *
 *  Approach 2: Dutch National flag algo
 *  1. Define 3 pointers: low, mid, high
 *  2. Rules:
 *      1. 0 to low - 1 : all zeros
 *      2. low to mid - 1: all ones
 *      3. mid to high: unsorted array
 *      4. high + 1 to n - 1: all tows
 *
 *      Notes:
 *      1: 0 to mid - 1: sorted containing 0,1
 *      2: high-1 to n-1: sorted containing 2
 *      3: Imp: mid to high is unsorted array so initially mid = 0 and high  = n - 1
 *
 *
 *      0   0   0   0   1   1   1   1   2  0  1  0  2  2  0  2  2  2  2  2
 *      |               |               |                 |              |
 *      0              low             mid              high            n - 1
 *
 */

fun sortColors(nums: IntArray): Unit {
    var low = 0 // 0 to low - 1 == 0's
    var mid  = 0 // low to mid - 1 == 1's
    var high = nums.size - 1 // high + 1 to n - 1 == 2's

    // imp: mid - high unsorted array
    while(mid < high) {
        if(nums[mid] == 0) {
            // swap low and mid
            swap(nums, mid, low)
            mid++
            low++

        } else if(nums[mid] == 1) {
            mid++

        } else if(nums[mid] == 2) {
            // swap mid and high
            swap(nums, mid, high)
            high --
        }
    }
}


fun swap(arr: IntArray, i: Int, j: Int) {
    val temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp
}


/**
 * problem: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
 *
 * Approach: Sliding Window
 *  1. Initial window is buying price = arr[0] and selling price = arr[1]
 *  2. We keep updating selling price to next element in array in loop
 *  3. update buying price if we get a lower price that current buying price
 *  4. Keep track of profit in each step and calculate max profit
 *
 *  Note: Here we are keep track of min element in arr[0] to arr[i] and use that as buying price as we want the min buying
 *  price for max profit
 *
 *  Time Complexity: O(n)
 *  Space Complexity: O(1)
 */
fun maxProfit(prices: IntArray): Int {
    var max_profit = 0
    // buying prices
    var bp = prices[0]

    for (i in 1 until prices.size) {
        val sp = prices[i]
        val temp_profit = sp - bp
        max_profit = maxOf(max_profit, temp_profit)

        // update buying prices
        if (sp < bp) {
            bp = sp
        }
    }
    return max_profit
}


// TODO - char array vs map complexity
/**
 * Problem: https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
 *
 * Approach: Sliding window
 *  1. Start l and r from 0
 *  2. we will keep map to keep track of char we visited: map: Char --> last position of char
 *  3. if s[r] is not in map --> we have not seen this char --> expand window --> r ++ --> put char in map
 *  4. If s[r] is in map and in window --> we encounter dup char --> our window break here
 *          --> calculate max --> update l --> start window after last position of s[r]
 *
 *  Note: In point 4, we want to check if any char found in map is in my current window or not
 *      ex. if s[1] = p and l = 3, r = 8 and s[8] = p, should I consider s[8] as dup?? No because s[1] is not in my window, my window start from 3
 *
 *
 *  Time Complexity: O(n), we are traversing string once
 *  Space Complexity: O(k), where k = longest substring, but we can say O(n) in case of no dup char
 *
 *
 * Ref: https://www.youtube.com/watch?v=-zSxTJkcdAo&ab_channel=takeUforward
 */
fun lengthOfLongestSubstring(s: String): Int {
    val map = mutableMapOf<Char, Int>()
    var l = 0
    var r = 0
    var max = 0
    val n = s.length

    while (r < n) {
        val ch = s[r]

        // check if we have seen this char window by checking if ch  in map and if yes it is between l and r
        val lastPosition = map.get(ch)
        if (lastPosition != null && lastPosition >= l) {
            // we have dup char so get length of current window l to r - 1
            max = maxOf(max, r - l)

            // move start of window after the last position of ch as we dont want that char in our window
            l = lastPosition + 1
        } else {
            map.put(ch, r)
            r ++
        }
    }

    // We are updating max only when we found dup char in loop, so to get max when no dup char in string
    max = maxOf(max, r - l)

    return max
}


/**
 * Problem: https://leetcode.com/problems/longest-repeating-character-replacement/submissions/1242645920/
 *
 * Approach:
 *  - Valid sub string:
 *      we can have max k replacement so a sub string is valid if it requires max k replacement to make all char same.
 *      How many replacement a substring need:  len(subStr) - count of max occuring char
 *      ex. for "BBBABA":  A -> 4, B -> 2, so we need 2 replacement
 *      len(BBBABA) = 6. maxF = 4 --> 6 - 4 = 2
 *
 *      So Valid SubStr :  len - maxF <= k
 *
 *  - sliding window: move r till we have valid substring. move l when current substring is not valid and start new substring
 *
 *
 *  Note:
 *  maxF: max frequency of char in substring. so it should increase when we move r but also decrease if we move l and move the char out of window.
 *  But we are not doing that BECAUSE if maxF = n it means at any point of time we have n same char in a window and len of that substring is already recorded
 *  now if we want to beat previous substring we need larger valid substring and hence maxF should be >= n, thats why in order to get the ans better that previous only
 *  we dont have to decrease maxF
 *

 * Ref: https://www.youtube.com/watch?v=gqXU1UyA8pk&ab_channel=NeetCode
 */
fun characterReplacement(s: String, k: Int): Int {
    var max = 0
    var l = 0
    val map = IntArray(26)
    var maxF = 0

    for (r in s.indices) {
        // update freq in map
        map[s[r] - 'A']++
        maxF = maxOf(maxF, map[s[r] - 'A'])

        // while substring is not valid keep updating l
        while ((r - l + 1) - maxF > k) {
            map[s[l] - 'A']--
            l++
        }

        max = max(max, r - l + 1)
    }

    return max
}


/**
 * problem: https://leetcode.com/problems/valid-parentheses/
 *
 * Approach: Stack
 *  1. If opening bracket push
 *  2, if closing bracket, corresponding opening must be at top of stack
 */
fun isValidParenthesis(s: String): Boolean {
    val stack = Stack<Char>()
    for (ch in s) {

        when (ch) {
            '[', '{', '(' -> {
                stack.push(ch)
            }
            ']' -> {
                if (stack.isEmpty() || stack.peek() != '[') return false else stack.pop()
            }
            '}' -> {
                if (stack.isEmpty() || stack.peek() != '{') return false else stack.pop()
            }
            ')' -> {
                if (stack.isEmpty() || stack.peek() != '(') return false else stack.pop()
            }

        }
    }
    return stack.empty()

}

/**
 * problem: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
 *
 * Approach:
 *  1. Draw a graph of input array.
 *      - Graph will look like a z
 *      - this graph will always have a dip/pivot (which is our answer)
 *      - graph will have 2 sorted parts
 *          1. 0 to pivot-1 : where each element is > end of array. call it SA1
 *          2. pivot to n-1:  where each element is less than first part and this part contains pivot(our ans). call it SA2
 *
 *   Note: our pivot is always less than its neighbour elements
 *   In case of mid =  0 left element is arr[n-1]. In case of mid = n - 1, right element is arr[0]
 *
 *   2. Find mid and check if it is our pivot by checking right and left element
 *   3. if arr[mid] > arr[end] : our mid is in SA1 so we need to move right
 *   4. if arr[mid] < arr[start]: out mid is in SA2. Since SA2 also contains pivot, can we say that this mid is not our ans? Yes because if this mid is pivot it
 *   would be caught in step 2 only
 *
 *   Note: Remember we have compare it with end of search space to decide where our mid is
 */
fun findMinInRotatedSortedArray(nums: IntArray): Int {

    // edge cases
    val n = nums.size
    if (nums[0] < nums.last()) return nums[0]
    if (nums.size == 1) return nums[0]


    var low = 0
    var high = n - 1

    while (low <= high) {
        val mid = low  + (high - low) / 2

        // We are using %, to avoid extra check for index 0 and n-1
        if (nums[mid] < nums[(mid + n - 1) % n] && nums[mid] < nums[(mid + 1) % n]) {
            // found pivot
            return nums[mid]
        } else if (nums[mid] > nums[high]) {
            // We are in SA1 so move right
            low = mid + 1
        } else {
            // We are in SA2 so move left
            high = mid - 1
        }
    }
    return -1
}

/**
 * Problem: https://leetcode.com/problems/search-in-rotated-sorted-array/description/
 *
 * Approach:
 *   1. Draw a graph of input array.
 *      - Graph will look like a z
 *      - this graph will always have a dip/pivot (which is our answer)
 *      - graph will have 2 sorted parts
 *          1. 0 to pivot-1 : where each element is > end of array. call it SA1
 *          2. pivot to n-1:  where each element is less than first part and this part contains pivot(our ans). call it SA2
 *
 *   2. find mid and return if this is target
 *   3. if mid is in SA1: elements < arr[mid] are present in SA1 and SA2 also, so we need one extra condition. if target if < arr[mid] and > arr[low] ,we are sure we need to move left other wise right
 *   4. if mid is in SA2: elements > arr[mid] are present in SA1 and SA2 also, so we need one extra condition. if target if > arr[mid] and < arr[high] ,we are sure we need to move right other wise left
 *
 */

fun searchInSOrtedRotatedArray(nums: IntArray, target: Int): Int {
    val n = nums.size
    var l = 0
    var r = n - 1

    while (l <= r) {
        val mid = l + (r - l)/2

        if (nums[mid] == target) {
            return mid

        }

        if (nums[mid] > nums[r]){
            // we are in SA1
            if (target < nums[mid] && target >= nums[l]) {
                r = mid - 1
            } else {
                l = mid + 1
            }
        } else {
            // We are in SA2
            if (target > nums[mid] && target <= nums[r]) {
                l = mid + 1
            } else {
                r = mid - 1
            }
        }
    }
    return  -1
}


/**
 * Problem: https://leetcode.com/problems/reverse-linked-list/
 *
 * Approach: Use multiple pointers to trash previous, current and next items
 *
 * Note: don't forget that after reversal the last node (head, before reversal) must point to NULL. So keep Start P with null only
 *
 * Time Complexity - O(n)
 * space Complexity - O(1)
 */
fun reverseLinkedList(head: ListNode?): ListNode? {
    if (head == null) return head

    var P: ListNode? = null
    var C = head
    var N = head?.next

    while (C != null) {
        C.next = P
        P = C
        C = N
        N = N?.next
    }

    return P
}

/**
 * Problem: https://leetcode.com/problems/merge-two-sorted-lists/submissions/1247491223/
 *
 * Approach: Loop over both list and compare element at each step
 *
 * Note: We Can use dummy head trick to avoid extra code to decide head. As we need to compare the head of both list and make the smaller one the head of ans
 * Note: Once one list is exhausted, we dont have to iterate over other list. Just point to the remaining part of the list. This is LL not array
 *
 */
fun mergeTwoSortedLinkedLists(list1: ListNode?, list2: ListNode?): ListNode? {
    // edge cases
    if (list1 == null) return list2
    if (list2 == null) return list1

    // Dummy head
    val preHead = ListNode(-1)
    var k = preHead
    var i = list1
    var j = list2

    while (i != null && j != null) {
        if (i.`val` < j.`val`) {
            k.next = i
            i = i.next
        } else {
            k.next = j
            j = j.next
        }
        k = k.next!!
    }

    // loop through rest of items
    k.next = i ?: j

    return preHead.next
}

/**
 * Problem: https://leetcode.com/problems/reorder-list/
 *
 * Approach:
 *  1. divide list in 2 half: Find mid using slow, fast pointer
 *  2. Reverse 2 half
 *  3. Merge both half
 *
 *  Ref: https://www.youtube.com/watch?v=S5bfdUTrKLM&t=563s&ab_channel=NeetCode
 */
fun reorderLinkedList(head: ListNode?): Unit {
    // Step 1: divide list in 2 parts
    var S = head
    var F = head?.next
    while (F?.next != null) {
        S = S?.next
        F = F.next?.next
    }

    // S.next is the 1st node of 2nd part
    // Lets reverse the 2nd part
    var P = S?.next
    var l2 = reverseLinkedList(P)

    S?.next = null  // s is the last node of 1st part of list so making point to null


    // Merge l1 and l2
    var l1 = head
    val preHead = ListNode(-1)
    var k: ListNode? = preHead
    while (l1 != null && l2 != null) {
        k?.next = l1
        l1 = l1.next
        k = k?.next

        k?.next = l2
        l2 = l2?.next
        k = k?.next
    }

    if (l1 != null) {
        k?.next  = l1
    }

    val ans = preHead.next
}

fun removeNthFromEnd(head: ListNode?, n: Int): ListNode? {
    var P = head
    var dis = n

    while (dis > 0 && P != null) {
        P = P?.next
        dis --
    }

    // Note: If P == null --> n >= size of list
    // If list size == n, then we just need to remove 1st nore i.e header
    if (P == null) return head?.next

    var C = head
    while (P?.next != null) {
        P = P.next
        C = C?.next
    }

    // Now C is pointing to the previous node of the target
    C?.next = C?.next?.next

    return head
}


/**
 * Problem: https://leetcode.com/problems/linked-list-cycle/submissions/1249027092/
 *
 * Approach: Slow and fast pointer
 * Note: Fast must move 2x then slow not 3x or 4x
 *
 * Ref: https://www.youtube.com/watch?v=wiOo4DC5GGA&ab_channel=takeUforward
 */
fun hasCycle(head: ListNode?): Boolean {
    var P = head
    var Q = head

    // If only one node and it points to null
    if (head?.next == null) return false

    while (P != null && Q != null) {
        P = P.next
        // Why Fast moves 2X, why not 3X or 4X???
        Q = Q.next?.next
        if (P == Q) return true
    }

    return false
}


/** Trees **/
fun levelOrder(root: TreeNode?): List<List<Int>> {

    val traversal = mutableListOf<List<Int>>()

    //edge case
    if (root == null) return traversal

    val queue = LinkedList<TreeNode?>().apply {
        add(root)
        add(null)
    }

    val level = mutableListOf<Int>()
    while (queue.isNotEmpty()) {
        val item = queue.pop()

        if (item == null) {
            // level completed so we need to add it in traversal
            traversal.add(level.toList())
            if (queue.isEmpty()) break
            queue.add(null)
            level.clear()
        } else {
            level.add(item.`val`)
            item?.left?.let {
                queue.add(it)
            }
            item?.right?.let {
                queue.add(it)
            }
        }

    }
    return traversal
}

/**
 * problem: https://leetcode.com/problems/maximum-depth-of-binary-tree/submissions/1265024123/
 *
 * Approach: Recursion in tree
 *
 * Time Complexity: O(n) to iterate all nodes
 * Space Complexity: O(n) in worst case where tree is skewed
 */
fun maxDepthOfBT(root: TreeNode?): Int {
    // base case
    if (root == null) return 0

    val leftHeight = maxDepthOfBT(root.left)
    val rightHeight = maxDepthOfBT(root.right)

    return maxOf(leftHeight, rightHeight) + 1
}


/**
 * Problem: https://leetcode.com/problems/binary-tree-maximum-path-sum/description/?source=submission-noac
 *
 * Approach: Recursion
 * Here maxPathSumHelp() is not returning max path sum rather it returns the longest sub path because that's we want to pass to the parent caller
 * and we keep track of max path sum in a separate variable
 *
 * Note: We are taking maxOf(0, maxPathSumHelp(root.left, maxSum)), instead of actual path Because, if right ot left subpath is -ve we should not take it
 *
 * Ref:https://www.youtube.com/watch?v=WszrfSwMz58&list=PLkjdNRgDmcc0Pom5erUBU4ZayeU9AyRRu&index=17&ab_channel=takeUforward
 */
fun maxPathSum(root: TreeNode?): Int {
    val maxSum = mutableListOf<Int>(Int.MIN_VALUE)
    maxPathSumHelp(root, maxSum)
    return maxSum.get(0)
}

fun maxPathSumHelp(root: TreeNode?, maxSum: MutableList<Int>): Int {

    // base case
    if (root == null) return 0

    // Note: We need to avoid -ve left or right path, if sub path gives -ve I will rather take 0
    val leftLongestPathSum = maxOf(0, maxPathSumHelp(root.left, maxSum))
    val rightLongestPathSum = maxOf(0, maxPathSumHelp(root.right, maxSum))

    // Find the max path sum passing through root and update if it is max
    val path = leftLongestPathSum + rightLongestPathSum + root.`val`
    maxSum[0] = maxOf(maxSum[0], path)

    // But to the parent we can't pass whole path we need to pass the longest branch
    return maxOf(leftLongestPathSum, rightLongestPathSum) + root.`val`
}


/**
 * Problem: https://leetcode.com/problems/same-tree/submissions/1266840605/
 *
 * Approach: Recursion
 *
 * Ref: https://leetcode.com/problems/same-tree/
 */
fun isSameTree(p: TreeNode?, q: TreeNode?): Boolean {
    //base case
    if (p == null || q == null) {
        return p == q
    }

    return (p.`val` == q.`val` && isSameTree(p.left, q.left) && isSameTree(p.right, q.right))
}

/**
 * Problem:
 * Approach: Recursion
 * Method returns the ancestor of the p,q or both OR null otherwise
 * For a node N
 *  - if its both subtree returns null --> p,q are not present in the tree with root N
 *  - if its both subtree returns non null --> p,q both are present in the tree with root N, i.e N is ancestor of both p and q
 *  - if one of the sub tree return non null --> N is one of the ancestor of p or q
 *
 *  Note: Please check the reference of path from root to node also to understand this problem
 *
 * Ref: https://www.youtube.com/watch?v=_-QHfMDde90&list=PLkjdNRgDmcc0Pom5erUBU4ZayeU9AyRRu&index=27&ab_channel=takeUforward
 * Ref: https://www.youtube.com/watch?v=fmflMqVOC7k&list=PLkjdNRgDmcc0Pom5erUBU4ZayeU9AyRRu&index=26&ab_channel=takeUforward
 */
fun lowestCommonAncestor(root: TreeNode?, p: TreeNode?, q: TreeNode?): TreeNode? {
    if (root?.`val` == null) return null

    if (root.`val` == p?.`val` || root.`val` == q?.`val`) {
        return root
    }

    val fromLeft = lowestCommonAncestor(root.left, p, q)
    val fromRight = lowestCommonAncestor(root.right, p, q)

    if (fromLeft != null && fromRight != null) {
        return root
    }

    return fromLeft ?: fromRight
}


/**
 * Problem: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/submissions/1268245820/
 * Approach: Recursion + BST properties
 * In BST right subtree > root > left subtree
 * so If both p and q
 *  - < root --> go to left sub tree to find ans
 *  - > root --> go to right sub tree to find ans
 *  - if one and smaller and one is larger it means root is the common ancestor
 */

fun lowestCommonAncestorBST(root: TreeNode?, p: TreeNode?, q: TreeNode?): TreeNode? {
    if (root?.`val` == null || p?.`val` == null || q?.`val` == null) return null

    if (p.`val` < root.`val` && q.`val` < root.`val`) return lowestCommonAncestorBST(root.left, p, q)
    if (p.`val` > root.`val` && q.`val` > root.`val`) return lowestCommonAncestorBST(root.right, p, q)

    return root
}


/**
 * Problem: https://leetcode.com/problems/invert-binary-tree/description/
 */
fun invertTree(root: TreeNode?): TreeNode? {
    if (root == null) return root
    if (root.left == null && root.right == null) return root
    val temp = root.left
    root.left = invertTree(root.right)
    root.right = invertTree(temp)
    return root
}


/**
 * Problem: https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/
 * Approach:
 * 1. first element of preorder is always root
 * 2. All elements at left of root elements in inorder traversal are part of left sub tree and on right are part of right sub tree.
 * i.e we can divide inorder array
 * - Use recursive approach with above pointers
 *
 * Ref: https://www.youtube.com/watch?v=aZNaLrVebKQ&ab_channel=takeUforward
 */
fun buildTree(preorder: IntArray, inorder: IntArray): TreeNode? {
    val map = mutableMapOf<Int, Int>()
    inorder.forEachIndexed { index, i ->
        map.put(i, index)
    }
    val root = build(inorder, 0, inorder.size - 1, preorder, 0, preorder.size  - 1,  map)
    return root
}

fun build( inorder: IntArray, ins: Int, ine: Int, preorder: IntArray, ps: Int, pe: Int, map: Map<Int, Int>): TreeNode? {
    if (ins > ine || ps > pe) return null
    val root = TreeNode(preorder[ps])

    val rootIndex = map[root.`val`] ?: 0
    val leftTreeElementsCount = rootIndex - ins
    val rightTreeElementsCount = ine - rootIndex

    root.left = if (leftTreeElementsCount == 0 )
        null
    else
        build(inorder, ins, rootIndex - 1, preorder, ps + 1, ps + leftTreeElementsCount, map)

    root.right = if (rightTreeElementsCount == 0 )
        null
    else
        build(inorder, rootIndex + 1, ine, preorder, ps + leftTreeElementsCount + 1, pe, map)

    return root
}


/**
 * Problem: https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/
 *
 * Approach: traverse the tree (BFS or DFS) and use any specific char for null nodes (children of leaf nodes) like #
 * - Can be done using Level oder or preorder traversal also
 * We are using level order traversal here
 *
 *
 * Ref:
 * Level order traversal: https://www.youtube.com/watch?v=-YbXySKJsX8&ab_channel=takeUforward
 * Pre order traversal: https://www.youtube.com/watch?v=u4JAi2JJhI8&ab_channel=NeetCode
 *
 */
// Encodes a URL to a shortened URL.
fun serialize(root: TreeNode?): String {
    val queue = LinkedList<TreeNode?>().apply {
        push(root)
    }
    var result = ""
    while (queue.isNotEmpty()) {
        val item = queue.pop()

        result += "${(item?.`val` ?: "#")},"


        if (item != null) {
            queue.add(item.left)
            queue.add(item.right)
        }
    }
    return result
}

// Decodes your encoded data to tree.
fun deserialize(data: String): TreeNode? {
    if (data.isEmpty()) return null
    val strs = data.split(",")
    if (strs.isEmpty()) return null
    if (strs.first() == "#") return null

    val root = TreeNode(strs.first().toInt())
    val queue = LinkedList<TreeNode?>().apply {
        add(root)
    }
    var index = 1
    while (queue.isNotEmpty()) {
        val item = queue.pop()

        var left = strs.getOrNull(index++) ?: "#"
        if (left != "#") {
            item?.left = TreeNode(left.toInt())
            queue.add(item?.left)
        } else {
            item?.left = null
        }

        val right = strs.getOrNull(index++) ?: "#"
        if (right != "#") {
            item?.right = TreeNode(right.toInt())
            queue.add(item?.right)
        } else {
            item?.right = null
        }
    }
    return root
}


/**
 * Problem: https://leetcode.com/problems/kth-smallest-element-in-a-bst/submissions/1273896667/
 *
 * Approach: Inorder traversal in BST is always sorted, so just count the node in inroder traversal and save kth value
 *
 * Ref: https://www.youtube.com/watch?v=9TJYWh0adfk&t=317s&ab_channel=takeUforward
 */
fun kthSmallest(root: TreeNode?, k: Int): Int {
    val ans = IntArray(size = 1)
    val counter = intArrayOf(0)
    kthSmallestHelper(root, counter, ans, k)
    return ans.get(0)
}

fun kthSmallestHelper(root: TreeNode?, counter: IntArray, ans: IntArray, k: Int) {
    if (root == null) return
    // Inorder traversal
    kthSmallestHelper(root.left, counter, ans, k)
    // increase the counter and check if this is the ans
    counter[0] = counter[0] + 1
    if (counter[0] == k) {
        ans[0] = root.`val`
    }
    kthSmallestHelper(root.right, counter, ans, k)
}


/**
 *  Problem: https://leetcode.com/problems/validate-binary-search-tree/description/
 *  Approach:
 *  1. Define a range of each node and cheeck if node falls in that range and so its left and right sub tree
 *  2. Do it recursive
 *  Note: Initial value for node is MAX and MIN, We took Long here to avoid Int max edge case
 *
 *
 *  Ref:https://www.youtube.com/watch?v=f-sj7I5oXEI&list=PLkjdNRgDmcc0Pom5erUBU4ZayeU9AyRRu&index=46&ab_channel=takeUforward
 */
fun isValidBST(root: TreeNode?): Boolean {
    return isValidBSTHelper(root, Long.MIN_VALUE, Long.MAX_VALUE)
}

fun isValidBSTHelper(root: TreeNode?, min: Long, max: Long): Boolean {
    if (root == null) return true

    return root.`val` in (min + 1) until max
            && isValidBSTHelper(root.left, min = min, max = root.`val`.toLong())
            && isValidBSTHelper(root.right, min = root.`val`.toLong(), max = max)
}

/**
 * Problem: https://leetcode.com/problems/subtree-of-another-tree/description/
 *
 * Approach:
 *  T is sub tree of S if
 *   1. T == S
 *   2. T is equal to left or right sub tree of S
 * - Recursive
 *
 */
fun isSubtree(root: TreeNode?, subRoot: TreeNode?): Boolean {

    if (root == null) return false

    if (isSameTree(root, subRoot)) return true

    return isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot)
}


fun main() {
    val root = TreeNode(2147483647)
    /* root.left = TreeNode(1)
     root.right = TreeNode(4)
     root.right?.left = TreeNode(3)
     root.left?.right = TreeNode(6)*/


    //val sr = serialize(root)
    print(isValidBST(root))
}



fun test(k: List<Int>): Int {
    val fmap = mutableMapOf<Int, Int>()
    for (i in k) {
        fmap[i] = fmap.getOrDefault(i, 0) + 1
    }

    var sum = 0
    fmap.forEach { key, value ->
        if (value == 1) {
            sum = sum + key
        }
    }

    return sum

}

fun allSubSet(k: List<Int>): Int {
    val allSubSets = mutableListOf<List<Int>>()
    val n = k.size

    k.forEachIndexed { index, i ->
        val start = i
        var end = i + 2
        while (end < n) {
            val subset = k.subList(start, end)
            allSubSets.add(subset)
            end = end + 2
        }
    }

    var sum = 0
    allSubSets.forEach {
        sum = sum + it.sum()
    }

    return sum
}
