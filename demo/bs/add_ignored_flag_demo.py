import matplotlib.pyplot as plt
import geopandas

import bstool


if __name__ == '__main__':
    shp_file = './data/buildchange/v0/shanghai/merged_shp/L18_106968_219320.shp'
    ignore_file = './data/buildchange/v0/shanghai/anno_v2/L18_106968_219320.png'
    geo_file = './data/buildchange/v0/shanghai/geo_info/L18_106968_219320.png'
    rgb_file = './data/buildchange/v0/shanghai/images/L18_106968_219320.jpg'

    objects = bstool.shp_parse(shp_file=shp_file,
                                geo_file=geo_file,
                                src_coord='4326',
                                dst_coord='pixel')
    origin_polygons = [obj['polygon'] for obj in objects]
    origin_properties = [obj['property'] for obj in objects]

    objects = bstool.mask_parse(ignore_file, subclasses=255)
    ignore_polygons = [obj['polygon'] for obj in objects]

    ignored_polygon_indexes = bstool.get_ignored_polygon_idx(origin_polygons, ignore_polygons)

    origin_properties = bstool.add_ignore_flag_in_property(origin_properties, ignored_polygon_indexes)

    # codes for visualization
    converted_polygons = []
    for origin_polygon, origin_property in zip(origin_polygons, origin_properties):
        if origin_property['ignore'] == 1:
            continue
        else:
            converted_polygons.append(origin_polygon)

    fig, ax = plt.subplots(1, 2)

    origin_polygons = geopandas.GeoSeries(origin_polygons)
    ignore_polygons = geopandas.GeoSeries(ignore_polygons)

    origin_df = geopandas.GeoDataFrame({'geometry': origin_polygons, 'foot_df':range(len(origin_polygons))})
    ignore_df = geopandas.GeoDataFrame({'geometry': ignore_polygons, 'ignore_df':range(len(ignore_polygons))})

    ignore_df.plot(ax=ax[0], color='red')
    origin_df.plot(ax=ax[0], facecolor='none', edgecolor='k')
    ax[0].set_title('Before ignoring')

    converted_polygons = geopandas.GeoSeries(converted_polygons)
    convert_df = geopandas.GeoDataFrame({'geometry': converted_polygons, 'foot_df':range(len(converted_polygons))})

    ignore_df.plot(ax=ax[1], color='red')
    convert_df.plot(ax=ax[1], facecolor='none', edgecolor='k')
    ax[1].set_title('After ignoring')

    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    plt.show()