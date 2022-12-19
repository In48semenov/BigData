#!/usr/bin/python3

import sys


def main(separator='\t'):
    for idx, line in enumerate(sys.stdin):
        chunk_size_curr, mean_curr = line.split(separator)
        chunk_size_curr, mean_curr = int(chunk_size_curr), float(mean_curr)
        if idx == 0:
            chunk_size = chunk_size_curr
            mean_value = mean_curr
        else:
            mean_value = (chunk_size * mean_value + chunk_size_curr * mean_curr) / (
                        chunk_size + chunk_size_curr)
            chunk_size += chunk_size_curr

    print(mean_value)


if __name__ == "__main__":
    main()
