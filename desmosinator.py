from PIL import Image
import numpy as np
from scipy.ndimage import convolve
import math
import potrace

V_KERNEL = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
H_KERNEL = V_KERNEL.transpose()


def main():
    source_image = "sea_otter.jpg"
    output_name = "sea_otter.txt"
    # load image
    with Image.open(f"images/{source_image}").convert("RGB") as im: # L changes the "mode" to 8-bit integer
        im.show()
        print(im.mode)
    
    # process image
    edged_image = detect_edges(im.transpose(Image.Transpose.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT), 900)
    cleaned_image = clean_up_edges(edged_image, 5, 5)

    #get_equations(cleaned_image, output_name)


def detect_edges(im, sensitivity):
    print("detecting edges!")
    imwidth, imheight = im.size
    image = np.array(im.getdata()).reshape(imheight, imwidth, 3) # height and width are reversed because reshape(A,B) returns an array with A rows of B columns
    edged_image = np.zeros(image.shape)

    for color_channel in range(3): # I used chatgpt to find scipy.ndimage.convolve, which makes things a lot faster (even though my old method works fine)
        h_edges = convolve(image[:, :, color_channel], H_KERNEL)
        v_edges = convolve(image[:, :, color_channel], V_KERNEL)
        edged_image[:,:,color_channel] = np.sqrt(h_edges ** 2 + v_edges ** 2)
        print(f"color_channel {color_channel} done!")
    
    print("edge detection done!")

    colored_test = Image.fromarray(edged_image.astype("uint8"), 'RGB')
    colored_test.show()

    # post-processing
    edged_image = reduce_to_bitmap(edged_image, sensitivity)

    test = Image.fromarray(edged_image)
    test.show()

    return edged_image


def reduce_to_bitmap(image, sensitivity):
    # get sum
    sum = np.sum(image, axis=2)
    sum[:] = np.where(sum > sensitivity, 255, 0) # this basically says, for each element in the np array: element = (element > 127) ? 1 : 0
    return sum


def clean_up_edges(edged_image, kernel_size, sensitivity):
    print("cleaning!")
    # pad the image
    CLEAN_KERNEL = np.ones((kernel_size, kernel_size))

    heatmap = convolve(edged_image, CLEAN_KERNEL, mode="constant", cval=0)
    mask = heatmap > sensitivity * 255
    clean_image = mask * edged_image

    print("cleaning done!")
    test = Image.fromarray(clean_image)
    test.show()
    return clean_image


def get_equations(edged_image, output_name):
    bmp = potrace.Bitmap(edged_image)
    print("tracing...")
    path = bmp.trace()
    print("traced!")
    output_file_path = f"outputs/{output_name}"
    open(output_file_path, "w").close() # wipe the file initially
    for curve in path:
        with open(output_file_path, 'a') as output_file:
            for segment_idx in range(len(curve)):
                segment = curve[segment_idx]
                end_point_x, end_point_y = segment.end_point.x, segment.end_point.y # NOTE: points have an x and y attribute, and the tutorial is bs
                # get start point
                if segment_idx == 0:
                    start_point_x, start_point_y = curve.start_point.x, curve.start_point.y
                else:
                    start_point_x, start_point_y = curve[segment_idx - 1].end_point.x, curve[segment_idx - 1].end_point.y

                if segment.is_corner: # we're not gonna do corners lol
                    c_x, c_y = segment.c.x, segment.c.y
                    output_file.write(get_line_between_points(start_point_x, start_point_y, c_x, c_y))
                    output_file.write(get_line_between_points(c_x, c_y, end_point_x, end_point_y))
                else:
                    c1_x, c1_y = segment.c1.x, segment.c1.y
                    c2_x, c2_y = segment.c2.x, segment.c2.y
                    x0, y0 = start_point_x, start_point_y
                    x1, y1 = c1_x, c1_y
                    x2, y2 = c2_x, c2_y
                    x3, y3 = end_point_x, end_point_y
                    # in potrace:
                    # ({x0}, {y0}) = either Curve.start_point or previous BezierSegment.end_point
                    # ({x1}, {y1}) = c1
                    # ({x2}, {y2}) = c2
                    # ({x3}, {y3}) = end_point
                    output_file.write(f"((1-t)((1-t)((1-t){x0}+t{x1})+t((1-t){x1}+t{x2}))+t((1-t)((1-t){x1}+t{x2})+t((1-t){x2}+t{x3})),(1-t)((1-t)((1-t){y0}+t{y1})+t((1-t){y1}+t{y2}))+t((1-t)((1-t){y1}+t{y2})+t((1-t){y2}+t{y3})))\n")


def get_line_between_points(x1, y1, x2, y2):
    if abs(x1 - x2) < 0.0000000001: # if it's very small such that it would be use e
        # a couple reasons why this looks cursed:
        # - to output "{hi!}", you have to do "{{hi!}}"
        # when copy+pasting from desmos, y=x{2<x<5} -> y=x\left\{2<x<5\right\}. but now you have to escape the "\"s with "\\", so you get \\{{\\}} and \\left and \\right
        equation = "x={x1:.10f} \\left\\{{{y1:.10f} < y < {y2:.10f}\\right\\}}\n".format( 
            x1=x1, 
            y1=y1 if (y1 < y2) else y2,
            y2=y1 if (y1 > y2) else y2
        )
        return equation
    else:
        m = (y2 - y1) / (x2 - x1)
        equation = "y-{y1:.10f}={m:.10f}(x-{x1:.10f}) \\left\\{{{start:.10f} < x < {end:.10f}\\right\\}}\n".format(
            y1=y1, m=m, x1=x1,
            start=x1 if (x1 < x2) else x2,
            end=x1 if (x1 > x2) else x2
        )
        return equation


main()