# visualization.py
"""
Visualization functions for satellite simulation: static plot and animation.
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
from shapely.geometry import box, MultiPolygon
from shapely.ops import unary_union
from matplotlib.patheffects import withStroke
from config import (
    reception_stations,
    use_custom_map,
    custom_map_path,
    custom_map_extent,
    animation_fps,
    animation_bitrate,
    fov_history_length,
    output_video_filename,
    latitude_limit,
)


def plot_static(satellites, sky_datetime_objects):
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    if use_custom_map:
        try:
            custom_map_image = plt.imread(custom_map_path)
            ax.imshow(
                custom_map_image,
                origin='upper',
                extent=custom_map_extent,
                transform=ccrs.PlateCarree(),
                zorder=0,
            )
        except FileNotFoundError:
            print(f"Custom map '{custom_map_path}' not found. Using stock image.")
            ax.stock_img(zorder=0)
    else:
        ax.stock_img(zorder=0)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='gray', zorder=1)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, zorder=2)
    map_boundary_box = box(-180, -90, 180, 90)

    for name, params in satellites.items():
        lons = params['longitudes']
        lats = params['latitudes']
        jump_mask = params['jump_mask']

        # plot ground tracks
        for i in range(len(lons) - 1):
            if not jump_mask[i+1]:
                ax.plot(
                    [lons[i], lons[i+1]],
                    [lats[i], lats[i+1]],
                    color=params['color'],
                    linewidth=1.5,
                    transform=ccrs.Geodetic(),
                    zorder=3,
                )

        # sample FOVs
        N_fov_samples = min(20, len(params['fov_polygons_shapely']))
        indices = list(range(0, len(params['fov_polygons_shapely']), max(1, len(params['fov_polygons_shapely']) // N_fov_samples)))
        for idx in indices:
            poly = params['fov_polygons_shapely'][idx]
            if poly and poly.is_valid and not poly.is_empty:
                vis = poly.intersection(map_boundary_box)
                if vis.is_empty:
                    continue
                if isinstance(vis, MultiPolygon):
                    for part in vis.geoms:
                        if part.is_valid and not part.is_empty:
                            x, y = part.exterior.xy
                            ax.plot(
                                x,
                                y,
                                color=params['color'],
                                alpha=0.3,
                                linewidth=1.0,
                                transform=ccrs.Geodetic(),
                                zorder=2,
                            )
                else:
                    x, y = vis.exterior.xy
                    ax.plot(
                        x,
                        y,
                        color=params['color'],
                        alpha=0.3,
                        linewidth=1.0,
                        transform=ccrs.Geodetic(),
                        zorder=2,
                    )

    # plot reception stations
    for st_name, st_info in reception_stations.items():
        ax.plot(
            st_info['lon'],
            st_info['lat'],
            marker=st_info['marker'],
            color=st_info['color'],
            markersize=10,
            transform=ccrs.Geodetic(),
            linestyle='',
            zorder=4,
        )
        ax.text(
            st_info['lon'] + 0.5,
            st_info['lat'] + 0.5,
            st_name,
            transform=ccrs.Geodetic(),
            color=st_info['color'],
            fontsize=9,
            zorder=5,
            path_effects=[withStroke(linewidth=2, foreground='white')],
        )

    plt.title("Satellite Ground Tracks & Sample FOVs")
    plt.tight_layout()
    plt.show()


def plot_coverage_overlay(satellites):
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    if use_custom_map:
        try:
            img = plt.imread(custom_map_path)
            ax.imshow(
                img,
                origin='upper',
                extent=custom_map_extent,
                transform=ccrs.PlateCarree(),
                zorder=0,
            )
        except FileNotFoundError:
            ax.stock_img(zorder=0)
    else:
        ax.stock_img(zorder=0)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='gray', zorder=1)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, zorder=2)

    # union of all FOVs clipped
    clip_box = box(-180, -latitude_limit, 180, latitude_limit)
    all_polys = []
    for params in satellites.values():
        for poly in params.get('fov_polygons_shapely', []):
            if poly and poly.is_valid and not poly.is_empty:
                clipped = poly.intersection(clip_box)
                if clipped.is_valid and not clipped.is_empty:
                    all_polys.append(clipped)

    if all_polys:
        coverage_union = unary_union(all_polys)
        if isinstance(coverage_union, MultiPolygon):
            geoms = coverage_union.geoms
        else:
            geoms = [coverage_union]

        for p in geoms:
            if p.is_valid and not p.is_empty:
                x, y = p.exterior.xy
                ax.fill(x, y, color='yellow', alpha=0.3, zorder=3, label='Covered Area')

    for st_name, st_info in reception_stations.items():
        ax.plot(
            st_info['lon'], st_info['lat'],
            marker=st_info['marker'], color=st_info['color'],
            markersize=10, transform=ccrs.Geodetic(), linestyle='', zorder=4
        )
        ax.text(
            st_info['lon'] + 0.5, st_info['lat'] + 0.5,
            st_name,
            transform=ccrs.Geodetic(), color=st_info['color'], fontsize=9, zorder=5,
            path_effects=[withStroke(linewidth=2, foreground='white')]
        )

    plt.title(f"Total Coverage Overlay ({latitude_limit}° to -{latitude_limit}°)")
    plt.tight_layout()
    plt.show()


def animate_coverage(satellites, sky_datetime_objects):
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    map_box = box(-180, -90, 180, 90)
    clip_box = box(-180, -latitude_limit, 180, latitude_limit)
    num_steps = len(sky_datetime_objects)

    # Counters: day land/ocean only
    day_land = 0
    day_ocean = 0

    # classify overlap with land
    def classify_surface(poly):
        for land_geom in cfeature.LAND.geometries():
            if poly.intersects(land_geom):
                return 'land'
        return 'ocean'

    def animate(frame):
        nonlocal day_land, day_ocean
        ax.clear()
        ax.set_global()

        t = sky_datetime_objects[frame]
        # build nightshade feature and get its geometries
        ns = Nightshade(t, alpha=0)
        night_polys = list(ns.geometries())
        ax.add_feature(Nightshade(t, alpha=0.3), zorder=1)

        # background
        if use_custom_map:
            try:
                img = plt.imread(custom_map_path)
                ax.imshow(img, origin='upper', extent=custom_map_extent,
                          transform=ccrs.PlateCarree(), zorder=0)
            except FileNotFoundError:
                ax.stock_img(zorder=0)
        else:
            ax.stock_img(zorder=0)

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)
        ax.gridlines(draw_labels=False, color='lightgray', alpha=0.5, zorder=2)

        # cumulative coverage (optional)
        polys = []
        for params in satellites.values():
            for poly in params['fov_polygons_shapely'][:frame+1]:
                if poly and poly.is_valid and not poly.is_empty:
                    cp = poly.intersection(clip_box)
                    if cp.is_valid and not cp.is_empty:
                        polys.append(cp)
        if polys:
            cov = unary_union(polys)
            geoms = cov.geoms if isinstance(cov, MultiPolygon) else [cov]
            for g in geoms:
                x, y = g.exterior.xy
                ax.fill(x, y, color='yellow', alpha=0.2, zorder=3)

        # plot tracks & FOV count day photos
        for name, params in satellites.items():
            lons = params['longitudes']
            lats = params['latitudes']
            mask = params['jump_mask']
            # ground track
            for i in range(frame):
                if not mask[i+1]:
                    ax.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]],
                            color=params['color'], transform=ccrs.Geodetic(), zorder=4)
            # current FOV
            curr = params['fov_polygons_shapely'][frame]
            if curr and curr.is_valid and not curr.is_empty:
                vis = curr.intersection(map_box)
                parts = vis.geoms if isinstance(vis, MultiPolygon) else [vis]
                for part in parts:
                    x, y = part.exterior.xy
                    ax.plot(x, y, color=params['color'], alpha=0.7,
                            linewidth=1.5, transform=ccrs.Geodetic(), zorder=5)

                # count only if centroid not in any night polygon (i.e. it's day)
                cen = curr.centroid
                in_night = any(n.contains(cen) for n in night_polys)
                if not in_night:
                    surf = classify_surface(curr)
                    if surf == 'land': day_land += 1
                    else:           day_ocean += 1

        # reception stations
        for st in reception_stations.values():
            ax.plot(st['lon'], st['lat'], marker=st['marker'], color=st['color'],
                    markersize=8, transform=ccrs.Geodetic(), zorder=6)
        for st in reception_stations.values():
            ax.text(st['lon']+0.5, st['lat']+0.5, st['marker'],
                    transform=ccrs.Geodetic(), color=st['color'], fontsize=8,
                    zorder=7, path_effects=[withStroke(linewidth=1.5, foreground='white')])

        # display counters under map
        ax.text(
            0.5, -0.1,
            f"Day-Land: {day_land}, Day-Ocean: {day_ocean}",
            transform=ax.transAxes, ha='center', va='top', fontsize=12
        )

        ax.set_title(f"Satellite Simulation: {t.strftime('%Y-%m-%d %H:%M:%S')} UTC", fontsize=10)
        return []

    anim = FuncAnimation(fig, animate, frames=num_steps, interval=1000/animation_fps, blit=False)
    writer = FFMpegWriter(fps=animation_fps, metadata=dict(artist='Satellite Simulator'), bitrate=animation_bitrate)
    anim.save(output_video_filename, writer=writer)
    plt.close(fig)


def animate_modis_only(satellites, sky_datetime_objects, output_filename="modis_animation.mp4"):
    """
    Animate only the MODIS satellite: blue ground track, blue FOV, red square for satellite position.
    """
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    map_boundary_box = box(-180, -90, 180, 90)
    num_steps = len(sky_datetime_objects)
    # Find MODIS satellite key
    modis_key = None
    for k in satellites:
        if "modis" in k.lower():
            modis_key = k
            break
    if not modis_key:
        raise ValueError("MODIS satellite not found in satellites dictionary.")
    params = satellites[modis_key]
    def animate(frame_num):
        ax.clear()
        ax.set_global()
        if use_custom_map:
            try:
                custom_map_image = plt.imread(custom_map_path)
                ax.imshow(custom_map_image, origin='upper', extent=custom_map_extent, transform=ccrs.PlateCarree(), zorder=0)
            except FileNotFoundError:
                ax.stock_img(zorder=0)
        else:
            ax.stock_img(zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', zorder=1)
        ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, zorder=2, color='lightgray', alpha=0.5)
        current_sim_time = sky_datetime_objects[frame_num]
        # Plot ground track up to current frame
        lons_hist = params['longitudes'][:frame_num+1]
        lats_hist = params['latitudes'][:frame_num+1]
        jump_mask_hist = params['jump_mask'][:frame_num+1]
        for i in range(len(lons_hist) - 1):
            if not jump_mask_hist[i+1]:
                ax.plot([lons_hist[i], lons_hist[i+1]], [lats_hist[i], lats_hist[i+1]], color='blue', linewidth=2.0, transform=ccrs.Geodetic(), zorder=3)
        # Plot FOV (current frame)
        current_fov = params['fov_polygons_shapely'][frame_num]
        if current_fov and current_fov.is_valid and not current_fov.is_empty:
            visible_poly = current_fov.intersection(map_boundary_box)
            if not visible_poly.is_empty:
                if isinstance(visible_poly, MultiPolygon):
                    for p_part in visible_poly.geoms:
                        if p_part.is_valid and not p_part.is_empty:
                            x, y = p_part.exterior.xy
                            ax.fill(x, y, color='blue', alpha=0.3, zorder=4)
                else:
                    x, y = visible_poly.exterior.xy
                    ax.fill(x, y, color='blue', alpha=0.3, zorder=4)
        # Plot current satellite position as a red square
        sat_lon = params['longitudes'][frame_num]
        sat_lat = params['latitudes'][frame_num]
        ax.plot(sat_lon, sat_lat, marker='s', color='red', markersize=14, transform=ccrs.Geodetic(), linestyle='', zorder=5)
        # Plot reception stations (static)
        for st_name, st_info in reception_stations.items():
            ax.plot(st_info["lon"], st_info["lat"], marker=st_info["marker"], color=st_info["color"], markersize=8, transform=ccrs.Geodetic(), linestyle='', zorder=6)
            ax.text(st_info["lon"] + 0.5, st_info["lat"] + 0.5, st_name, transform=ccrs.Geodetic(), color=st_info["color"], fontsize=8, zorder=7, path_effects=[withStroke(linewidth=1.5, foreground='white')])
        # Update title with current time
        ax.set_title(f"MODIS Satellite Simulation: {current_sim_time.strftime('%Y-%m-%d %H:%M:%S')} UTC", fontsize=12)
        return []
    anim = FuncAnimation(fig, animate, frames=num_steps, interval=1000/animation_fps, blit=False)
    writer = FFMpegWriter(fps=animation_fps, metadata=dict(artist='Satellite Simulator'), bitrate=animation_bitrate)
    try:
        anim.save(output_filename, writer=writer)
        print(f"\nAnimation saved as {output_filename}")
    except Exception as e:
        print(f"\nError saving animation: {e}")
        print("Please ensure FFMpeg is installed and in your system's PATH.")
    plt.close(fig) 