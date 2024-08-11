export type RGBList = [number, number, number];

const isRGBArray = (item: RGBList | string | undefined): item is RGBList => {
  if ((item as RGBList).sort !== undefined) {
    if (
      (item as RGBList).length >= 3 &&
      (item as RGBList)[0] >= 0 &&
      (item as RGBList)[0] <= 255 &&
      (item as RGBList)[1] >= 0 &&
      (item as RGBList)[1] <= 255 &&
      (item as RGBList)[2] >= 0 &&
      (item as RGBList)[2] <= 255
    ) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
};

export function normalizeColor(
  color: RGBList | string | undefined,
  fallback: RGBList,
): {
  color_str_100_opacity: string;
  color_str_90_opacity: string;
  color_str_80_opacity: string;
  color_str_70_opacity: string;
  color_str_60_opacity: string;
  color_str_50_opacity: string;
  color_str_40_opacity: string;
  color_str_30_opacity: string;
  color_str_20_opacity: string;
  color_str_10_opacity: string;
} {
  let normalised_color = fallback;

  if (color) {
    if (isRGBArray(color)) {
      normalised_color = color;
    } else {
      if (/^#([0-9a-fA-F]{3})$/.test(color)) {
        normalised_color = [
          parseInt(`${color[1]}${color[1]}`, 16),
          parseInt(`${color[2]}${color[2]}`, 16),
          parseInt(`${color[3]}${color[3]}`, 16),
        ];
      } else if (/^#([0-9a-fA-F]{6})$/.test(color)) {
        normalised_color = [
          parseInt(color.slice(1, 3), 16),
          parseInt(color.slice(3, 5), 16),
          parseInt(color.slice(5, 7), 16),
        ];
      }
    }
  }

  return {
    color_str_100_opacity: `rgb(${normalised_color[0]} ${normalised_color[1]} ${normalised_color[2]})`,
    color_str_90_opacity: `rgb(${normalised_color[0]} ${normalised_color[1]} ${normalised_color[2]} / 0.9)`,
    color_str_80_opacity: `rgb(${normalised_color[0]} ${normalised_color[1]} ${normalised_color[2]} / 0.8)`,
    color_str_70_opacity: `rgb(${normalised_color[0]} ${normalised_color[1]} ${normalised_color[2]} / 0.7)`,
    color_str_60_opacity: `rgb(${normalised_color[0]} ${normalised_color[1]} ${normalised_color[2]} / 0.6)`,
    color_str_50_opacity: `rgb(${normalised_color[0]} ${normalised_color[1]} ${normalised_color[2]} / 0.5)`,
    color_str_40_opacity: `rgb(${normalised_color[0]} ${normalised_color[1]} ${normalised_color[2]} / 0.4)`,
    color_str_30_opacity: `rgb(${normalised_color[0]} ${normalised_color[1]} ${normalised_color[2]} / 0.3)`,
    color_str_20_opacity: `rgb(${normalised_color[0]} ${normalised_color[1]} ${normalised_color[2]} / 0.2)`,
    color_str_10_opacity: `rgb(${normalised_color[0]} ${normalised_color[1]} ${normalised_color[2]} / 0.1)`,
  };
}
