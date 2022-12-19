#!/usr/bin/python3

import sys
from csv import reader


def main(index_col_price: int = 9, separator: str = '\t'):
    mean_value, count_value = 0, 0
    prices = []
    for line in reader(sys.stdin):
        try:
            price = line[index_col_price]
            if price != "price":
                mean_value += float(price)
                count_value += 1
                prices.append(float(price))
        except:
            continue

    mean_value /= count_value
    var_value = sum([(price - mean_value) ** 2 for price in prices]) / (count_value - 1)
    print(
        count_value,
        separator,
        mean_value,
        separator,
        var_value
    )


if __name__ == "__main__":
    main()
