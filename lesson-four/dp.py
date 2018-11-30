#coding:utf8

def cal_edit_distance_with_cache(str1, str2, cache):
    if len(str1) == 0:
        return [len(str2), ["left insert '%s'" % str2 if len(str2) > 0 else ""]]
    elif len(str2) == 0:
        return [len(str1), ["righ insert '%s'" % str1 if len(str1) > 0 else ""]]

    concat_str = '%s_%s' % (str1, str2)
    if concat_str in cache:
        return cache[concat_str]

    case1_distance = cal_edit_distance(str1[1:], str2)
    case1_distance[0] += 1
    case1_distance[1].append("left remove '%c'" % str1[0])
    case2_distance = cal_edit_distance(str1, str2[1:])
    case2_distance[0] += 1
    case2_distance[1].append("right remove '%c'" % str2[0])
    case3_distance = cal_edit_distance(str1[1:], str2[1:])
    case3_distance[0] += (0 if str1[0] == str2[0] else 2)
    case3_distance[1].append("'%c' == '%c' " % (str1[0], str1[0]) if str1[0] == str2[0] else "left '%c' change to '%c'" % (str1[0], str2[0]))
    min_distance = min(case1_distance, case2_distance, case3_distance, key=lambda dist: dist[0])

    cache[concat_str] = min_distance
    return min_distance

def cal_edit_distance(str1, str2):
    cache = {}
    return cal_edit_distance_with_cache(str1, str2, cache)

distance, op = cal_edit_distance('ab', 'ad')

print(distance)
print(op[::-1])

print('=' * 100)

def cal_knapsack_with_cache(price_dict, k, length, cache):
    if length == 0:
        return (0, [])
    if length in cache:
        return cache[length]
    r = (k if  k <= length else length)
    result = []
    for i in range(r):
        left_price = price_dict[i+1] if i+1 in price_dict else 0
        right_price = cal_knapsack_with_cache(price_dict, k, length - i - 1, cache)
        result.append((left_price + right_price[0], right_price[1] + [i+1]))
    max_value = max(result, key=lambda price: price[0])
    cache[length] = max_value
    return max_value

def cal_knapsack(price_dict, length):
    cache = {}
    k = max(price_dict.keys())
    return cal_knapsack_with_cache(price_dict, k, length, cache)

price_dict = {1:1, 2:4, 3:4, 5:7}
total_price, split = cal_knapsack(price_dict, 6)
print(total_price)
print(split)

