from PIL import Image
import numpy as np
import math
import potrace

H_KERNEL = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])


V_KERNEL = H_KERNEL.transpose()


def main():
    source_image = "github_octocat.png"
    output_name = "octocat.txt"
    # load image
    with Image.open(f"images/{source_image}").convert("L") as im: # L changes the "mode" to 8-bit integer
        im.show()

    edged_image = detect_edges(im.transpose(method=Image.Transpose.ROTATE_180))
    cleaned_image = clean_up_edges(edged_image, 20, 25)

    get_equations(cleaned_image, output_name)


def detect_edges(im):
    print("detecting edges!")
    width, height = im.size
    image = np.array(im.getdata()).reshape(height, width) # height and width are reversed because reshape(A,B) returns an array with A rows of B columns
    edged_image = np.zeros(image.shape)
    for x in range(width):
        for y in range (height):
            if x != 0 and x != width-1 and y != 0 and y != height-1:
                subsection = image[y-1:y+2, x-1:x+2]
                edged_image[y, x] = calculate_gradient(subsection)
        if x % 100 == 0: # logging
            print(f"x = {x} is done")

    edged_image[:] = np.where(edged_image > 150, 255, 0) # this basically says, for each element in the np array: element = (element > 127) ? 255 : 0
    print("edge detection done!")
    test = Image.fromarray(edged_image).convert("1")
    test.show()
    return edged_image


def clean_up_edges(edged_image, kernel_size, sensitivity):
    print("cleaning!")
    # pad the image
    CLEAN_KERNEL = np.ones((kernel_size, kernel_size))
    padding = math.floor(CLEAN_KERNEL.shape[0] / 2)
    padded_image = np.pad(edged_image, padding)
    # go through and do the clean-up
    height, width = edged_image.shape
    clean_image = padded_image
    for x in range(width):
        for y in range(height):
            if (not (x < padding or x > width - padding or y < padding or y > height - padding)):
                subsection = padded_image[y-padding:y+padding+1, x-padding:x+padding+1]
                if np.sum(subsection) <= sensitivity*255: # 3 is arbitrary
                    clean_image[y, x] = 0
        if x % 100 == 0: # logging
            print(f"x = {x} is done!")
    # get rid of padding
    image_to_return = clean_image[padding:height-padding, padding:width-padding]
    print("cleaning done!")
    test = Image.fromarray(image_to_return).convert("1")
    test.show()
    return clean_image


def calculate_gradient(subsection):
    gradient_x = np.sum(subsection * H_KERNEL)
    gradient_y = np.sum(subsection * V_KERNEL)
    return math.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)


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
    if x1 == x2:
        # a couple reasons why this looks cursed:
        # - to output "{hi!}", you have to do "{{hi!}}"
        # when copy+pasting from desmos, y=x{2<x<5} -> y=x\left\{2<x<5\right\}. but now you have to escape the "\"s with "\\", so you get \\{{\\}} and \\left and \\right
        equation = "x={x1} \\left\\{{{y1} < y < {y2}\\right\\}}\n".format( 
            x1=x1, 
            y1=y1 if (y1 < y2) else y2,
            y2=y1 if (y1 > y2) else y2
        )
        return equation
    else:
        m = (y2 - y1) / (x2 - x1)
        equation = "y-{y1}={m}(x-{x1}) \\left\\{{{start} < x < {end}\\right\\}}\n".format(
            y1=y1, m=m, x1=x1,
            start=x1 if (x1 < x2) else x2,
            end=x1 if (x1 > x2) else x2
        )
        return equation


main()