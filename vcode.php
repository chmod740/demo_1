<?php
        /****************************************
        *  ��֤�� v0.9
        *  Powerd by awaysoft.com
        *  ���������GPLv3����
        *  2011-07-15
        ****************************************/
        /* $len Ϊ����ַ������ȣ�$typeΪ���ͣ�aΪ�ַ����֣�cΪ�ַ�,nΪ���֣� �Ѿ�ȥ�������󵼵��ַ� */
        function get_rand_string($len=4, $type="n"){
                if ($len < 0) $len = 4;
                if ($type == 'a') $chars = 'ABCDEFGHJKMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz023456789';
                else if ($type == 'c') $chars = 'ABCDEFGHJKMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz';
                else if ($type == 'n') $chars = '0123456789';
                else $chars = '0123456789';

                $result = '';
                for ($i = 0; $i < $len; $i ++){
                        $index = mt_rand(0, strlen($chars) - 1);
                        $result .= substr($chars, $index, 1);
                }
                return $result;
        }

        session_start();

        /* ���ͼƬ���ͣ��ַ����ȣ����� */
        $imgtype = 'gif';
        $len = 4;
        $vcodetype = 'n';

        $width = 15 * $len;
        $height = 24;
        /* ��������ַ�����д��SESSION */
        $vcode = get_rand_string($len, $vcodetype);
        $_SESSION['vcode'] = $vcode;
        header("Content-type: image/".$imgtype);

        if($imgtype != 'gif' && function_exists('imagecreatetruecolor')){
                $im = imagecreatetruecolor($width, $height);
        }else{
                $im = imagecreate($width, $height);
        }

        $r = mt_rand(0, 255);
        $g = mt_rand(0, 255);
        $b = mt_rand(0, 255);
        /* ���ɱ�����ɫ */
        $backColor = ImageColorAllocate($im, $r, $g, $b);
        /* ���ɱ߿���ɫ */
        $borderColor = ImageColorAllocate($im, 0, 0, 0);
        /* ���ɸ��ŵ���ɫ */
        $pointColor = ImageColorAllocate($im, mt_rand(0, 255), mt_rand(0, 255), mt_rand(0, 255));

        /* ����λ�� */
        imagefilledrectangle($im, 0, 0, $width - 1, $height - 1, $backColor);
        /* �߿�λ�� */
        imagerectangle($im, 0, 0, $width - 1, $height - 1, $borderColor);

        /* �ַ�����ɫ(������ɫ) */
        $stringColor = ImageColorAllocate($im, 255 - $r, 255 - $g, 255 - $b);

        /* �������ŵ� */
        $pointNumber = mt_rand($len * 25, $len * 50);
        for($i=0; $i<=$pointNumber; $i++){
                $pointX = mt_rand(2,$width-2);
                $pointY = mt_rand(2,$height-2);
                imagesetpixel($im, $pointX, $pointY, $pointColor);
        }

        imagettftext($im, 15, 0, 4, 20, $stringColor, "include/Vera.ttf", $vcode);
        $image_out = 'Image' . $imgtype;
        $image_out($im);
        @ImageDestroy($im);
?>
