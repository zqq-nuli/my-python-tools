import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


DEBUG = True

def analyze_template(template):
    hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_template, lower_yellow, upper_yellow)
    
    yellow_hsv = hsv_template[yellow_mask > 0]
    
    if yellow_hsv.size > 0:
        h_mean, h_std = np.mean(yellow_hsv[:, 0]), np.std(yellow_hsv[:, 0])
        s_mean, s_std = np.mean(yellow_hsv[:, 1]), np.std(yellow_hsv[:, 1])
        v_mean, v_std = np.mean(yellow_hsv[:, 2]), np.std(yellow_hsv[:, 2])
    else:
        h_mean, h_std = 25, 5
        s_mean, s_std = 200, 50
        v_mean, v_std = 200, 50
    
    yellow_area = np.sum(yellow_mask > 0)
    
    if DEBUG:
        print(f"Template analysis results:")
        print(f"H mean: {h_mean:.2f}, std: {h_std:.2f}")
        print(f"S mean: {s_mean:.2f}, std: {s_std:.2f}")
        print(f"V mean: {v_mean:.2f}, std: {v_std:.2f}")
        print(f"Yellow area: {yellow_area} pixels")
    
    return (h_mean, h_std), (s_mean, s_std), (v_mean, v_std), yellow_area, yellow_mask

def detect_target_in_complex_background(image, template):
    (h_mean, h_std), (s_mean, s_std), (v_mean, v_std), template_yellow_area, template_mask = analyze_template(template)
    
    h_range = (max(0, h_mean - h_std * 2), min(180, h_mean + h_std * 2))
    s_range = (max(0, s_mean - s_std), 255)
    v_range = (max(0, v_mean - v_std), 255)
    
    if DEBUG:
        print(f"HSV ranges for detection:")
        print(f"H range: {h_range}")
        print(f"S range: {s_range}")
        print(f"V range: {v_range}")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([h_range[0], s_range[0], v_range[0]])
    upper_bound = np.array([h_range[1], s_range[1], v_range[1]])
    color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Number of contours found: {len(contours)}")
    
    possible_targets = []
    rejected_contours = {
        'too_small': [],
        'too_large': [],
        'wrong_aspect_ratio': [],
        'low_extent': []
    }
    
    # 定义阈值
    thresholds = {
        'min_area': 0,
        'max_area': template_yellow_area * 2,
        'min_aspect_ratio': 0.5,
        'max_aspect_ratio': 2.0,
        'min_extent': 0
    }

    # 根据阈值筛选出可能包含目标的contours
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        rect_area = w * h
        extent = float(area) / rect_area
        
        contour_info = {
            'position': (x, y, w, h),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'extent': extent
        }

        if area < thresholds['min_area']:
            rejected_contours['too_small'].append(contour_info)
        elif area > thresholds['max_area']:
            rejected_contours['too_large'].append(contour_info)
        elif aspect_ratio < thresholds['min_aspect_ratio'] or aspect_ratio > thresholds['max_aspect_ratio']:
            rejected_contours['wrong_aspect_ratio'].append(contour_info)
        elif extent < thresholds['min_extent']:
            rejected_contours['low_extent'].append(contour_info)
        else:
            roi = image[y:y+h, x:x+w]
            possible_targets.append((x, y, w, h, roi))
            if DEBUG:
                print(f"Possible target {len(possible_targets)}: Position (x={x}, y={y}), Size (w={w}, h={h}), Area={area:.2f}, Aspect Ratio={aspect_ratio:.2f}, Extent={extent:.2f}")

    if DEBUG:
        print(f"Number of possible targets: {len(possible_targets)}")
        for reason, contours in rejected_contours.items():
            print(f"Number of contours rejected due to {reason}: {len(contours)}")
    
    
    best_match = None
    best_score = -np.inf
    
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, template_thresh = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY)
    template_contours, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    match_results = []
    
    # 形状检测及模板检测
    for i, (x, y, w, h, roi) in enumerate(possible_targets):
        resized_template = cv2.resize(template, (w, h))
        res = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_thresh = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY)
        roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if roi_contours and template_contours:
            shape_score = cv2.matchShapes(roi_contours[0], template_contours[0], cv2.CONTOURS_MATCH_I2, 0)
            shape_score = 1 / (1 + shape_score)
        else:
            shape_score = 0
        
        # 分数定义
        score = max_val * 0.6 + shape_score * 0.4
        match_results.append((x, y, w, h, score, max_val, shape_score))
        
        print(f"Target {i+1} scores: Template match = {max_val:.4f}, Shape match = {shape_score:.4f}, Total = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_match = (x, y, w, h)
    
    if best_match:
        print(f"Best match: Position (x={best_match[0]}, y={best_match[1]}), Size (w={best_match[2]}, h={best_match[3]}), Score={best_score:.4f}")
    else:
        print("No match found")
    
    return best_match, best_score, color_mask, template_mask, match_results, rejected_contours, thresholds

def visualize_results(image, template, best_match, score, color_mask, template_mask, match_results, rejected_contours, thresholds):
    plt.figure(figsize=(20, 20))
    
    # 最终检测结果
    plt.subplot(321)
    result = image.copy()
    if best_match:
        x, y, w, h = best_match
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f'Final Detection Result (Score: {score:.2f})')
    plt.axis('off')
    
    # 所有contours分类
    plt.subplot(322)
    contours_img = image.copy()
    colors = {
        'possible': (0, 255, 0),  # 绿色
        'too_small': (255, 0, 0),  # 蓝色
        'too_large': (0, 0, 255),  # 红色
        'wrong_aspect_ratio': (255, 255, 0),  # 青色
        'low_extent': (255, 0, 255)  # 洋红色
    }
    for x, y, w, h, _, _, _ in match_results:
        cv2.rectangle(contours_img, (x, y), (x+w, y+h), colors['possible'], 2)
    for reason, contours in rejected_contours.items():
        for contour_info in contours:
            x, y, w, h = contour_info['position']
            cv2.rectangle(contours_img, (x, y), (x+w, y+h), colors[reason], 2)
    plt.imshow(cv2.cvtColor(contours_img, cv2.COLOR_BGR2RGB))
    plt.title('All Contours Classification')
    plt.axis('off')

    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=[c/255 for c in color], edgecolor='none') for color in colors.values()]
    plt.legend(legend_elements, colors.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 颜色掩码
    plt.subplot(323)
    plt.imshow(color_mask, cmap='gray')
    plt.title('Color Mask')
    plt.axis('off')
    
    # 模板匹配分数热图
    plt.subplot(324)
    template_scores = np.zeros_like(image[:,:,0], dtype=float)
    for x, y, w, h, _, template_score, _ in match_results:
        template_scores[y:y+h, x:x+w] = template_score
    plt.imshow(template_scores, cmap='hot', interpolation='nearest')
    plt.title('Template Matching Scores')
    plt.colorbar()
    plt.axis('off')
    
    # 形状匹配分数热图
    plt.subplot(325)
    shape_scores = np.zeros_like(image[:,:,0], dtype=float)
    for x, y, w, h, _, _, shape_score in match_results:
        shape_scores[y:y+h, x:x+w] = shape_score
    plt.imshow(shape_scores, cmap='hot', interpolation='nearest')
    plt.title('Shape Matching Scores')
    plt.colorbar()
    plt.axis('off')
    
    # 总分数热图
    plt.subplot(326)
    total_scores = np.zeros_like(image[:,:,0], dtype=float)
    for x, y, w, h, total_score, _, _ in match_results:
        total_scores[y:y+h, x:x+w] = total_score
    plt.imshow(total_scores, cmap='hot', interpolation='nearest')
    plt.title('Total Scores')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    

    # 打印未匹配原因信息和得分
    print("\nContour Rejection Summary:")
    for reason, contours in rejected_contours.items():
        print(f"\n{reason}: {len(contours)} contours")
        if reason == 'too_small':
            print(f"Threshold: minimum area = {thresholds['min_area']}")
        elif reason == 'too_large':
            print(f"Threshold: maximum area = {thresholds['max_area']}")
        elif reason == 'wrong_aspect_ratio':
            print(f"Threshold: aspect ratio range = {thresholds['min_aspect_ratio']} to {thresholds['max_aspect_ratio']}")
        elif reason == 'low_extent':
            print(f"Threshold: minimum extent = {thresholds['min_extent']}")
        
        for i, contour_info in enumerate(contours[:5], 1):
            x, y, w, h = contour_info['position']
            print(f"  {i}. Position: (x={x}, y={y}), Size: (w={w}, h={h})")
            print(f"     Area: {contour_info['area']:.2f}")
            print(f"     Aspect Ratio: {contour_info['aspect_ratio']:.2f}")
            print(f"     Extent: {contour_info['extent']:.2f}")
        if len(contours) > 5:
            print(f"  ... and {len(contours) - 5} more")


# 待检测图片文件夹路径
image_file_path = 'data/target'
# 目标模板图片路径
target_mask_file_path = 'data/pubg_target2.jpg'
for file in os.listdir(image_file_path):
    if file.endswith('.png'):
        image = cv2.imread(f'{image_file_path}/{file}')
        template = cv2.imread(target_mask_file_path)
        best_match, score, color_mask, template_mask, match_results, rejected_contours, thresholds = detect_target_in_complex_background(image, template)
        visualize_results(image, template, best_match, score, color_mask, template_mask, match_results, rejected_contours, thresholds)





import cv2
import numpy as np


def adaptive_color_threshold(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    adaptive_thresh = cv2.adaptiveThreshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        199,
        1,
    )
    combined_mask = cv2.bitwise_and(
        cv2.inRange(hsv, lower_yellow, upper_yellow), adaptive_thresh
    )
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print(x, y, w, h)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return result


cv2.imshow("result", adaptive_color_threshold("1234.png"))
cv2.waitKey(0)
cv2.destroyAllWindows()
