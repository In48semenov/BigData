#!/usr/bin/python3

import sys
from csv import reader


def main(index_col_price: int = 9, separator: str = '\t'):
    sum_price, count_value = 0, 0
    for line in reader(sys.stdin):
        try:
            price = line[index_col_price]
            if price != "price":
                sum_price += float(price)
                count_value += 1
        except:
            continue

    print(
        count_value,
        separator,
        sum_price / count_value
    )


if __name__ == "__main__":
    main()
