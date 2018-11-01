import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class Inpainter():
    def __init__(self, image, mask, patch_size=9, diff_algorithm='sq', plot_progress=False):
        self.image = image.astype('uint8')
        self.mask = mask.astype('uint8')
        # 进行光滑处理消除噪声
        self.mask = cv.GaussianBlur(self.mask, (3, 3), 1.5)
        self.mask = (self.mask > 0).astype('uint8')
        self.fill_image = np.copy(self.image)
        self.fill_range = np.copy(self.mask)
        self.patch_size = patch_size
        # 信誉度
        self.confidence = (self.mask == 0).astype('float')
        self.height = self.mask.shape[0]
        self.width = self.mask.shape[1]
        self.total_fill_pixel = self.fill_range.sum()

        self.diff_algorithm = diff_algorithm
        self.plot_progress = plot_progress
        # 初始化成员变量

        # 边界矩阵
        self.front = None
        self.D = None
        # 优先级
        self.priority = None
        # 边界等照度线
        self.isophote = None
        # 目标点
        self.target_point = None
        # 灰度图片
        self.gray_image = None

    def inpaint(self):
        while self.fill_range.sum() != 0:
            self._get_front()
            self.gray_image = cv.cvtColor(
                self.fill_image, cv.COLOR_RGB2GRAY).astype('float')/255
            self._log()

            if self.plot_progress:
                self._plot_image()

            self._update_priority()
            target_point = self._get_target_point()
            self.target_point = target_point
            best_patch_range = self._get_best_patch_range(target_point)
            self._fill_image(target_point, best_patch_range)

        return self.fill_image

    # 打印日志

    def _log(self):
        progress_rate = 1-self.fill_range.sum()/self.total_fill_pixel
        progress_rate *= 100
        print('填充进度为%.2f' % progress_rate, '%')

    # 动态显示图片更新情况
    def _plot_image(self):
        fill_range = 1-self.fill_range
        fill_range = fill_range[:, :, np.newaxis].repeat(3, axis=2)

        image = self.fill_image*fill_range

        # 空洞填充为白色
        white_reginon = (self.fill_range-self.front)*255
        white_reginon = white_reginon[:, :, np.newaxis].repeat(3, axis=2)
        image += white_reginon

        plt.clf()
        plt.imshow(image)
        plt.draw()
        plt.pause(0.001)

    # 填充图片
    def _fill_image(self, target_point, source_patch_range):
        target_patch_range = self._get_patch_range(target_point)
        # 获取待填充点的位置
        fill_point_positions = np.where(self._patch_data(
            self.fill_range, target_patch_range) > 0)

        # 更新填充点的信誉度
        target_confidence = self._patch_data(
            self.confidence, target_patch_range)
        target_confidence[fill_point_positions[0], fill_point_positions[1]] =\
            self.confidence[target_point[0], target_point[1]]

        # 更新待填充点像素
        source_patch = self._patch_data(self.fill_image, source_patch_range)
        target_patch = self._patch_data(self.fill_image, target_patch_range)
        target_patch[fill_point_positions[0], fill_point_positions[1]] =\
            source_patch[fill_point_positions[0], fill_point_positions[1]]

        # 更新剩余填充点
        target_fill_range = self._patch_data(
            self.fill_range, target_patch_range)
        target_fill_range[:] = 0

    # 获取最佳匹配图片块的范围
    def _get_best_patch_range(self, template_point):
        diff_method_name = '_'+self.diff_algorithm+'_diff'
        diff_method = getattr(self, diff_method_name)

        template_patch_range = self._get_patch_range(template_point)
        patch_height = template_patch_range[0][1]-template_patch_range[0][0]
        patch_width = template_patch_range[1][1]-template_patch_range[1][0]

        best_patch_range = None
        best_diff = float('inf')
        lab_image = cv.cvtColor(self.fill_image, cv.COLOR_RGB2Lab)
        # lab_image=np.copy(self.fill_image)

        for x in range(self.height-patch_height+1):
            for y in range(self.width-patch_width+1):
                source_patch_range = [
                    [x, x+patch_height],
                    [y, y+patch_width]
                ]
                if self._patch_data(self.fill_range, source_patch_range).sum() != 0:
                    continue
                diff = diff_method(
                    lab_image, template_patch_range, source_patch_range)

                if diff < best_diff:
                    best_diff = diff
                    best_patch_range = source_patch_range

        return best_patch_range

    # 使用平方差比较算法计算两个区域的区别
    def _sq_diff(self, img, template_patch_range, source_patch_range):
        mask = 1-self._patch_data(self.fill_range, template_patch_range)
        mask = mask[:, :, np.newaxis].repeat(3, axis=2)
        template_patch = self._patch_data(img, template_patch_range)*mask
        source_patch = self._patch_data(img, source_patch_range)*mask

        return ((template_patch-source_patch)**2).sum()

    # 加入欧拉距离作为考量
    def _sq_with_eucldean_diff(self, img, template_patch_range, source_patch_range):
        sq_diff = self._sq_diff(img, template_patch_range, source_patch_range)
        eucldean_distance = np.sqrt((template_patch_range[0][0]-source_patch_range[0][0])**2 +
                                    (template_patch_range[1][0]-source_patch_range[1][0])**2)
        return sq_diff+eucldean_distance

    def _sq_with_gradient_diff(self, img, template_patch_range, source_patch_range):
        sq_diff = self._sq_diff(img, template_patch_range, source_patch_range)
        target_isophote = np.copy(
            self.isophote[self.target_point[0], self.target_point[1]])
        target_isophote_val = np.sqrt(
            target_isophote[0]**2+target_isophote[1]**2)
        gray_source_patch = self._patch_data(self.gray_image, source_patch_range)
        source_patch_gradient = np.nan_to_num(np.gradient(gray_source_patch))
        source_patch_val = np.sqrt(
            source_patch_gradient[0]**2+source_patch_gradient[1]**2)
        patch_max_pos = np.unravel_index(
            source_patch_val.argmax(),
            source_patch_val.shape
        )
        source_isophote = np.array([-source_patch_gradient[1, patch_max_pos[0], patch_max_pos[1]],
                                    source_patch_gradient[0, patch_max_pos[0], patch_max_pos[1]]])
        source_isophote_val = source_patch_val.max()

        # 计算两者之间的cos(theta)
        dot_product = abs(
            source_isophote[0]*target_isophote[0]+source_isophote[1] * target_isophote[1])
        norm = source_isophote_val*target_isophote_val
        cos_theta = 0
        if norm != 0:
            cos_theta = dot_product/norm
        val_diff = abs(source_isophote_val-target_isophote_val)
        return sq_diff-cos_theta+val_diff

    # 获取目标点的位置

    def _get_target_point(self):
        return np.unravel_index(self.priority.argmax(), self.priority.shape)

    # 使用Laplace算子求边界
    def _get_front(self):
        self.front = (cv.Laplacian(self.fill_range, -1) > 0).astype('uint8')

    def _update_priority(self):
        self._update_front_confidence()
        self._update_D()
        self.priority = self.confidence*self.D*self.front

    # 更新D
    def _update_D(self):
        normal = self._get_normal()
        isophote = self._get_isophote()
        self.isophote = isophote
        self.D = abs(normal[:, :, 0]*isophote[:, :, 0]**2 +
                     normal[:, :, 1]*isophote[:, :, 1]**2)+0.001
    # 更新边界点的信誉度

    def _update_front_confidence(self):
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch_range = self._get_patch_range(point)
            sum_patch_confidence = self._patch_data(
                self.confidence, patch_range).sum()
            area = (patch_range[0][1]-patch_range[0][0]) * \
                (patch_range[1][1]-patch_range[1][0])
            new_confidence[point[0], point[1]] = sum_patch_confidence/area

        self.confidence = new_confidence

    # 获取边界上法线的单位向量
    def _get_normal(self):
        x_normal = cv.Scharr(self.fill_range, cv.CV_64F, 1, 0)
        y_normal = cv.Scharr(self.fill_range, cv.CV_64F, 0, 1)
        normal = np.dstack([x_normal, y_normal])
        norm = np.sqrt(x_normal**2+y_normal**2).reshape(self.height,
                                                        self.width, 1).repeat(2, axis=2)
        norm[norm == 0] = 1
        unit_normal = normal/norm
        return unit_normal

    # 获取patch周围的等照度线
    def _get_isophote(self):
        gray_image = np.copy(self.gray_image)
        gray_image[self.fill_range == 1] = None
        gradient = np.nan_to_num(np.array(np.gradient(gray_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = np.zeros([self.height, self.width, 2])
        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch_range(point)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)
            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )
            # 旋转90度
            max_gradient[point[0], point[1], 0] = \
                -patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = \
                patch_x_gradient[patch_max_pos]

        return max_gradient

    # 获取图片块的范围
    def _get_patch_range(self, point):
        half_patch_size = (self.patch_size-1)//2
        patch_range = [
            [
                max(0, point[0]-half_patch_size),
                min(point[0]+half_patch_size+1, self.height)
            ],
            [
                max(0, point[1]-half_patch_size),
                min(point[1]+half_patch_size+1, self.width)
            ]
        ]
        return patch_range

    # 获取patch中的数据
    @staticmethod
    def _patch_data(img, patch_range):
        return img[patch_range[0][0]:patch_range[0][1], patch_range[1][0]:patch_range[1][1]]
