#!/bin/bash

signal_file="signal.txt"
# 定义要运行的命令列表
commands=(
    "nohup python3 -m clrs.examples.run --algorithms mst_prim > mst_prim.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms articulation_points > articulation_points.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms activity_selector > activity_selector.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms bellman_ford > bellman_ford.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms bfs > bfs.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms binary_search > binary_search.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms bridges > bridges.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms bubble_sort > bubble_sort.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms dag_shortest_paths > dag_shortest_paths.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms dfs > dfs.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms dijkstra > dijkstra.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms find_maximum_subarray_kadane > find_maximum_subarray_kadane.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms floyd_warshall > floyd_warshall.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms graham_scan > graham_scan.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms heapsort > heapsort.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms insertion_sort > insertion_sort.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms jarvis_march > jarvis_march.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms kmp_matcher > kmp_matcher.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms lcs_length > lcs_length.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms matrix_chain_order > matrix_chain_order.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms minimum > minimum.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms mst_kruskal > mst_kruskal.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms naive_string_matcher > naive_string_matcher.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms optimal_bst > optimal_bst.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms quickselect > quickselect.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms quicksort > quicksort.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms segments_intersect > segments_intersect.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms strongly_connected_components > strongly_connected_components.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms task_scheduling > task_scheduling.txt 2>&1"
    "nohup python3 -m clrs.examples.run --algorithms topological_sort > topological_sort.txt 2>&1"
)

# 循环遍历命令列表
for cmd in "${commands[@]}"
do
#    signal_file=".signal_file"
    
    # 检查是否存在信号量文件，如果存在则等待
    while [ -f "$signal_file" ]; do sleep 1; done

    # 执行当前命令
    eval $cmd

    # 创建信号量文件，表示当前命令已完成
    touch "$signal_file"

    # 删除信号量文件，以便后续命令可以执行
    rm "$signal_file"
done

# 将脚本与当前终端分离
disown -a