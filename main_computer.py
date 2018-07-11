#!/usr/bin/env python

from computer import Computer

def plot_peaks():
    index = [list(range(0, 10)),
             list(range(140, 150)),
             list(range(90, 100))]

    for block in index:
        print(block)
        for country_id in block:
            print("\t", computer.gt_parser.get_country_by_id(country_id))

if __name__ == "__main__":
    computer = Computer()

    # Plot histogram for country
    # computer.plot_histogram_for_column(column_name="country",
    #                                    bins=100,
    #                                    xlabel="Country Id",
    #                                    ylabel="Frequency",
    #                                    info_threshold=3)

    # Plot geographic map (heatmap)
    computer.plot_geographical_heatmap(filename="geomap.png",
                                       heatmap=False)



    # Plot peaks
    # plot_peaks()

    knn = computer.compute_knn_on_location()
