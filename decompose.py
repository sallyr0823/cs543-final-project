import os
import sys
import argparse
import sys
sys.path.append("/root/Digital_MakeUp_Face_Generation/Codes")
from FaceDetector import FaceExtractor
import cv2



if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from krahenbuhl2013.krahenbuhl2013 import DenseCRF
except ImportError:
    print("")
    print("Error: cannot import 'krahenbuhl2013'.")
    print("")
    print("This is a custom C++ extension and can be compiled with:")
    print("")
    print(("    cd %s" % os.path.join(os.path.dirname(os.path.abspath(__file__)), 'krahenbuhl2013')))
    print("    make")
    sys.exit(1)

from solver import IntrinsicSolver
from input import IntrinsicInput
from params import IntrinsicParameters
import image_util


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Decompose an image using the algorithm presented in:\n'
            '    Sean Bell, Kavita Bala, Noah Snavely. "Intrinsic Images in the Wild".\n'
            '    ACM Transactions on Graphics (SIGGRAPH 2014).\n'
            '    http://intrinsic.cs.cornell.edu.\n'
            '\n'
            'The output is rescaled for viewing and encoded as sRGB PNG images'
            '(unless --linear is specified).'
        )
    )

    parser.add_argument(
        'input', metavar='<file>', help='Input image')

    parser.add_argument(
        '-r', '--reflectance', metavar='<file>',
        help='Reflectance layer output name (saved as sRGB image)', required=False)

    parser.add_argument(
        '-s', '--shading', metavar='<file>',
        help='Shading layer output name (saved as sRGB image)', required=False)

    parser.add_argument(
        '-m', '--mask', metavar='<file>', type=str,
        help='Mask filename', required=False, default=None)

    parser.add_argument(
        '-j', '--judgements', metavar='<file>',
        help='Judgements file from the Intrinsic Images in the Wild dataset', required=False, default=None)

    parser.add_argument(
        '-p', '--parameters', metavar='<file>',
        help='Parameters file (JSON format).  See params.py for a list of parameters.', required=False, default=None)

    parser.add_argument(
        '-l', '--linear', action='store_true',
        help='if specified, assume input is linear and generate linear output, otherwise assume input is sRGB and generate sRGB output', required=False)

    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help="if specified, don't print logging info", required=False)

    parser.add_argument(
        '--show-labels', action='store_true',
        help="if specified, also output labels", required=False)

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

    # obtain arguments
    args = parser.parse_args()
    image_filename = args.input
    base, _ = os.path.splitext(image_filename)
    r_filename = args.reflectance if args.reflectance else base + '-r.png'
    s_filename = args.shading if args.shading else base + '-s.png'
    mask_filename = args.mask
    judgements_filename = args.judgements
    parameters_filename = args.parameters
    sRGB = not args.linear
    if not r_filename.endswith('.png'):
        r_filename += '.png'
    if not s_filename.endswith('.png'):
        s_filename += '.png'

    print('Input:')
    print(('  image_filename:', image_filename))
    print(('  mask_filename:', mask_filename))
    print(('  judgements_filename:', judgements_filename))
    print(('  parameters_filename:', parameters_filename))
    print('Output:')
    print(('  r_filename:', r_filename))
    print(('  s_filename:', s_filename))


    sourceImage = cv2.imread(image_filename)
    f = FaceExtractor()
    # output_image1, feature_points1, triangulation1,K1 = FeatureDetection.landmark_detection(sourceImage)
    output_image1, feature_points1, triangulation1,K1 = f.landmark_pipeline(sourceImage)
    
    cv2.imwrite("temp_output.png",output_image1)

    input = IntrinsicInput.from_file(
        "temp_output.png",
        image_is_srgb=sRGB,
        mask_filename=mask_filename,
        judgements_filename=judgements_filename,
    )


    # input = IntrinsicInput.from_array(
    # makeUpLightness_3channel, 
    # image_is_srgb=False,  
    # mask_filename=mask_filename,
    # judgements_filename = judgements_filename,
    # )

    print(('mask_nnz: %s' % input.mask_nnz))
    print(('rows * cols: %s' % (input.rows * input.cols)))

    # load parameters
    if parameters_filename:
        params = IntrinsicParameters.from_file(parameters_filename)
    else:
        params = IntrinsicParameters()

    params.logging = not args.quiet

    # solve
    solver = IntrinsicSolver(input, params)
    r, s, decomposition = solver.solve()
    print("reach here")
    # save output
    image_util.save(r_filename, r, mask_nz=input.mask_nz, rescale=True, srgb=sRGB)
    image_util.save(s_filename, s, mask_nz=input.mask_nz, rescale=True, srgb=sRGB)
    if args.show_labels:
        labels_vis = decomposition.get_labels_visualization()
        r_path, r_ext = os.path.splitext(r_filename)
        image_util.save('%s_labels%s' % (r_path, r_ext), labels_vis, mask_nz=solver.input.mask_nz, rescale=True)

    # compute error
    if judgements_filename:
        print(('WHDR: %.1f%%' % (input.compute_whdr(r) * 100.0)))
