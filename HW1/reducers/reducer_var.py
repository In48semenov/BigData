#!/usr/bin/python3

import sys


def main(separator='\t'):
    for idx, line in enumerate(sys.stdin):
        chunk_size_curr, mean_curr, var_curr = line.split(separator)
        chunk_size_curr, mean_curr, var_curr = int(chunk_size_curr), float(mean_curr), float(var_curr)
        if idx == 0:
            chunk_size = chunk_size_curr
            mean_value = mean_curr
            var_value = var_curr
        else:
            var_value = (chunk_size * var_value + chunk_size_curr * var_curr) / (
                        chunk_size + chunk_size_curr) + chunk_size * chunk_size_curr * (
                                    (mean_value - mean_curr) / (chunk_size + chunk_size_curr)) ** 2
            mean_value = (chunk_size * mean_value + chunk_size_curr * mean_curr) / (
                    chunk_size + chunk_size_curr)
            chunk_size += chunk_size_curr

    print(var_value)


if __name__ == "__main__":
    main()
