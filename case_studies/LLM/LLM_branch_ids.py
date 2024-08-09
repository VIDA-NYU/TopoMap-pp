branch_ids = {
    1: [45, 68, 104, 146, 195, 377, 462, 501, 525, 545, 584, 610, 697, 792, 864, 1019, 1098, 1100, 1121, 1229, 1364, 1454, 1476, 1489, 1515, 1691, 1692, 1721, 1861, 1933, 1943, 1977, 2026, 2030, 2053, 2136, 2140, 2182, 2237, 2310, 2323, 2375, 2425, 2570, 2595, 2656, 2684, 2720, 2835, 2853, 2904, 2927, 2975, 3093, 3373, 3390, 3439, 3445, 3589, 3669, 3691, 3777, 3822, 3867, 3921, 4030, 4046, 4080, 4093, 4201, 4235, 4324, 4325, 4464, 4467, 4472, 4503, 4546, 4559, 4569, 4643, 4652, 4691, 4723, 4808, 4862, 4871, 4912, 4917, 5098, 5103, 5127, 5135, 5201, 5283, 5284, 5296, 5379, 5388, 5397, 5462, 5541, 5544, 5554, 5644, 5770, 5809, 5824, 5845, 5846, 5889, 5923, 5948, 6177, 6228, 6285, 6286, 6296, 6314, 6375, 6402, 6404, 6418, 6435, 6492, 6511, 6600, 6664, 6669, 18, 72, 125, 137, 156, 193, 197, 274, 280, 372, 409, 503, 539, 550, 593, 640, 653, 680, 695, 713, 799, 878, 881, 887, 916, 940, 951, 969, 1006, 1038, 1062, 1091, 1101, 1129, 1166, 1170, 1193, 1204, 1299, 1308, 1348, 1366, 1399, 1429, 1494, 1549, 1550, 1564, 1575, 1639, 1667, 1698, 1713, 1742, 1767, 1790, 1839, 1840, 1842, 1853, 1866, 1895, 1896, 1951, 1964, 1973, 1999, 2035, 2096, 2112, 2122, 2159, 2207, 2253, 2278, 2283, 2301, 2350, 2431, 2445, 2448, 2455, 2456, 2469, 2471, 2477, 2481, 2541, 2551, 2569, 2591, 2617, 2627, 2638, 2659, 2675, 2698, 2700, 2743, 2748, 2752, 2759, 2813, 2894, 2922, 2950, 2968, 2976, 3062, 3067, 3109, 3125, 3126, 3140, 3175, 3223, 3242, 3257, 3289, 3338, 3366, 3388, 3398, 3407, 3410, 3411, 3423, 3426, 3485, 3494, 3500, 3559, 3564, 3571, 3596, 3611, 3617, 3620, 3639, 3700, 3715, 3716, 3724, 3770, 3871, 3895, 3915, 3927, 3936, 3945, 3947, 3954, 3957, 3964, 3985, 3994, 3997, 4013, 4053, 4065, 4157, 4163, 4172, 4183, 4205, 4217, 4228, 4256, 4266, 4314, 4322, 4323, 4326, 4334, 4375, 4398, 4401, 4441, 4443, 4453, 4469, 4479, 4486, 4508, 4525, 4556, 4574, 4575, 4625, 4631, 4737, 4764, 4772, 4775, 4813, 4855, 4890, 4910, 4928, 4930, 4943, 4955, 4962, 4972, 4975, 5010, 5012, 5026, 5038, 5069, 5078, 5080, 5082, 5090, 5111, 5114, 5172, 5180, 5192, 5198, 5212, 5220, 5233, 5240, 5242, 5258, 5271, 5274, 5275, 5280, 5316, 5334, 5357, 5386, 5400, 5470, 5480, 5484, 5485, 5516, 5519, 5523, 5527, 5546, 5547, 5639, 5643, 5672, 5694, 5793, 5832, 5844, 5855, 5863, 5875, 5943, 5954, 5995, 5996, 6013, 6024, 6051, 6067, 6109, 6151, 6178, 6226, 6257, 6297, 6318, 6452, 6494, 6519, 6526, 6578, 6580, 6581, 6643, 6645, 6652, 412, 476, 668, 768, 831, 836, 1145, 1455, 1498, 1504, 1749, 1900, 2418, 2422, 2436, 2545, 2583, 2658, 2674, 2688, 2692, 2734, 2758, 2831, 3197, 3306, 3530, 3643, 3737, 3740, 4238, 4403, 4567, 4595, 4693, 4732, 4781, 4794, 4812, 4875, 4948, 5049, 5077, 5344, 5467, 5494, 5563, 5577, 5605, 5638, 5664, 5801, 5805, 5815, 5820, 5839, 5852, 5878, 5880, 5970, 6086, 6107, 6130, 6157, 6180, 6189, 6209, 6217, 6229, 6238, 6243, 6259, 6265, 6266, 6280, 6335, 6346, 6353, 6390, 6423, 6442, 6448, 6504, 6513, 6541, 6593, 6604, 6638, 6640, 6650, 278, 313, 325, 352, 414, 435, 464, 651, 747, 808, 820, 857, 970, 1139, 1271, 1325, 1611, 1724, 1777, 1818, 1864, 1919, 2065, 2272, 2300, 2339, 2357, 2711, 2778, 2850, 2863, 3049, 3130, 3141, 3176, 3254, 3455, 3563, 3613, 3754, 3778, 3813, 4012, 4015, 4215, 4218, 4239, 4357, 4457, 4500, 4696, 4796, 4843, 4858, 4860, 4988, 4998, 5000, 5022, 5027, 5193, 5208, 5219, 5267, 5300, 5340, 5507, 5540, 5583, 5649, 5684, 5700, 5712, 5831, 5997, 5998, 6021, 6061, 6074, 6211, 6230, 6251, 6295, 6304, 6319, 6352, 6462, 6493, 6499, 6570, 6584, 6617, 6660, 6670],
    2: [23, 91, 160, 226, 240, 271, 273, 345, 483, 605, 729, 841, 927, 944, 1040, 1110, 1240, 1312, 1380, 1400, 1467, 1486, 1605, 1654, 1821, 1841, 1909, 1948, 2042, 2113, 2236, 2238, 2414, 2565, 2664, 2715, 2772, 2802, 2844, 2932, 3044, 3083, 3245, 3283, 3414, 3540, 3710, 3846, 3855, 3891, 3951, 4004, 4484, 4533, 4536, 4539, 4564, 4629, 4699, 4712, 4743, 4800, 4889, 4939, 4979, 5083, 5144, 5184, 5378, 5439, 5619, 5626, 5750, 5829, 5850, 5867, 5895, 5987, 6104, 6142, 6160, 6299, 6400, 6509, 6591, 6596, 6656, 12, 31, 39, 44, 53, 62, 70, 99, 147, 187, 233, 255, 256, 269, 270, 283, 317, 384, 411, 421, 495, 526, 547, 556, 574, 599, 611, 705, 706, 743, 750, 767, 774, 783, 791, 794, 797, 809, 843, 880, 911, 943, 946, 963, 981, 990, 1054, 1057, 1067, 1082, 1111, 1117, 1135, 1138, 1155, 1160, 1179, 1201, 1209, 1210, 1213, 1220, 1235, 1242, 1248, 1255, 1300, 1301, 1334, 1335, 1336, 1359, 1390, 1424, 1425, 1433, 1451, 1488, 1512, 1519, 1531, 1543, 1576, 1577, 1579, 1580, 1581, 1599, 1625, 1626, 1630, 1637, 1682, 1683, 1708, 1719, 1757, 1761, 1783, 1815, 1833, 1845, 1882, 1908, 1913, 1915, 1962, 1971, 1982, 1985, 1989, 1995, 2012, 2022, 2045, 2080, 2088, 2093, 2117, 2142, 2160, 2163, 2174, 2188, 2205, 2225, 2243, 2255, 2259, 2260, 2321, 2329, 2335, 2340, 2359, 2373, 2393, 2400, 2413, 2453, 2465, 2478, 2498, 2528, 2543, 2563, 2566, 2572, 2581, 2582, 2639, 2644, 2650, 2695, 2706, 2724, 2726, 2773, 2814, 2820, 2823, 2854, 2856, 2857, 2872, 2877, 2881, 2887, 2902, 2913, 2914, 2952, 2974, 2990, 3009, 3011, 3034, 3068, 3081, 3128, 3144, 3145, 3151, 3194, 3195, 3206, 3221, 3249, 3250, 3268, 3273, 3292, 3308, 3317, 3320, 3324, 3331, 3370, 3400, 3416, 3424, 3437, 3459, 3473, 3491, 3502, 3516, 3544, 3548, 3560, 3606, 3637, 3659, 3660, 3673, 3696, 3702, 3709, 3712, 3714, 3721, 3726, 3751, 3775, 3776, 3784, 3801, 3803, 3835, 3844, 3845, 3849, 3862, 3869, 3896, 3913, 3975, 3995, 4002, 4022, 4025, 4035, 4040, 4045, 4058, 4062, 4067, 4096, 4115, 4119, 4146, 4165, 4169, 4174, 4175, 4181, 4185, 4221, 4232, 4241, 4246, 4248, 4249, 4310, 4341, 4387, 4402, 4404, 4416, 4421, 4426, 4455, 4473, 4480, 4505, 4506, 4514, 4515, 4519, 4554, 4580, 4584, 4604, 4630, 4639, 4697, 4716, 4728, 4859, 4872, 4874, 4879, 4887, 4947, 4963, 4967, 4992, 4997, 5015, 5033, 5043, 5047, 5054, 5060, 5122, 5146, 5147, 5161, 5185, 5194, 5251, 5299, 5308, 5341, 5355, 5359, 5360, 5366, 5382, 5383, 5418, 5421, 5496, 5504, 5512, 5518, 5542, 5551, 5557, 5609, 5616, 5655, 5656, 5678, 5734, 5752, 5795, 5800, 5814, 5817, 5848, 5884, 5897, 5930, 5931, 5951, 5965, 5989, 6015, 6039, 6105, 6111, 6127, 6133, 6146, 6149, 6167, 6198, 6281, 6287, 6293, 6315, 6387, 6431, 6440, 6466, 6470, 6564, 148, 182, 266, 480, 544, 630, 754, 826, 971, 1447, 1570, 1850, 1947, 2126, 2171, 2550, 2978, 3264, 3342, 3520, 3523, 3823, 4160, 4244, 4340, 4362, 4475, 4498, 4603, 4801, 5120, 5187, 5356, 5500, 5567, 5720, 5760, 5808, 5822, 5833, 5976, 5999, 6035, 6055, 6248, 6309, 6349, 6443, 6447, 6465, 6527, 6540, 6599, 6628, 6654, 6655, 6665, 165, 316, 433, 660, 947, 982, 1167, 1514, 1633, 2007, 2061, 2062, 2071, 2203, 2217, 2284, 2307, 2489, 2524, 2712, 2812, 3209, 3216, 3377, 3472, 3529, 3583, 3989, 4001, 4048, 4106, 4317, 4354, 4465, 4489, 4499, 4538, 4551, 4565, 4576, 4596, 4663, 4668, 4707, 4919, 4941, 4956, 5076, 5117, 5119, 5164, 5189, 5200, 5246, 5254, 5292, 5297, 5320, 5448, 5481, 5556, 5636, 5647, 5735, 5837, 5873, 5973, 5983, 5984, 6079, 6118, 6203, 6210, 6213, 6337, 6354, 6358, 6401, 6411, 6424, 6553, 6567, 6602],
    3: [1352, 2522, 4510, 6092, 6663, 0, 74, 83, 86, 139, 167, 180, 212, 213, 222, 259, 314, 383, 418, 529, 766, 778, 965, 1015, 1041, 1116, 1241, 1280, 1284, 1313, 1347, 1351, 1357, 1445, 1482, 1492, 1524, 1555, 1563, 1596, 1678, 1806, 1813, 1912, 1934, 1944, 2090, 2105, 2115, 2172, 2222, 2327, 2399, 2463, 2499, 2678, 2682, 2829, 2859, 2933, 2951, 2960, 3070, 3096, 3205, 3305, 3363, 3412, 3415, 3552, 3652, 3860, 3925, 4047, 4055, 4073, 4088, 4159, 4210, 4223, 4259, 4303, 4304, 4328, 4367, 4390, 4579, 4779, 5221, 5331, 5416, 5510, 5676, 5841, 6017, 6144, 6430, 5055, 5621, 1628, 3193, 3431, 3556, 4673, 5059, 5210, 5600, 6000],
    4: [27, 84, 357, 521, 1060, 1063, 1297, 2206, 2388, 2496, 3590, 3657, 3798, 4547, 4903, 4985, 5489, 5714, 5751, 5810, 5836, 5946, 6072, 6123, 6155, 6194, 6473, 6657, 7, 58, 59, 60, 77, 78, 92, 114, 118, 127, 155, 158, 174, 206, 219, 238, 244, 248, 260, 272, 284, 306, 315, 321, 335, 365, 371, 381, 398, 404, 442, 469, 491, 512, 532, 533, 570, 596, 598, 607, 612, 628, 632, 637, 664, 678, 724, 725, 740, 758, 780, 793, 804, 834, 846, 850, 854, 860, 875, 886, 892, 920, 928, 932, 959, 961, 968, 1012, 1029, 1089, 1109, 1130, 1185, 1198, 1230, 1268, 1309, 1310, 1321, 1327, 1342, 1353, 1356, 1372, 1413, 1450, 1502, 1503, 1516, 1535, 1552, 1597, 1601, 1606, 1619, 1650, 1677, 1701, 1753, 1756, 1758, 1759, 1768, 1778, 1791, 1822, 1827, 1881, 1916, 1926, 1931, 1959, 2013, 2032, 2037, 2041, 2044, 2086, 2114, 2125, 2134, 2141, 2150, 2166, 2186, 2218, 2221, 2276, 2285, 2308, 2311, 2313, 2345, 2368, 2380, 2387, 2415, 2486, 2513, 2588, 2733, 2749, 2765, 2774, 2815, 2824, 2838, 2875, 2896, 2908, 2912, 2930, 2949, 2963, 2996, 3024, 3117, 3135, 3166, 3226, 3230, 3259, 3284, 3290, 3346, 3498, 3558, 3580, 3615, 3624, 3632, 3655, 3687, 3766, 3818, 3853, 3944, 3949, 3961, 3990, 3993, 3998, 4016, 4037, 4193, 4199, 4321, 4371, 4378, 4427, 4513, 4524, 4529, 4561, 4589, 4593, 4620, 4665, 4695, 4700, 4718, 4725, 4730, 4756, 4828, 4924, 4984, 5018, 5052, 5058, 5075, 5139, 5156, 5157, 5169, 5175, 5178, 5225, 5236, 5270, 5295, 5302, 5319, 5338, 5352, 5370, 5391, 5411, 5463, 5466, 5472, 5508, 5543, 5581, 5635, 5716, 5738, 5925, 6003, 6008, 6140, 6306, 6371, 6376, 6396, 6406, 6506, 6563, 6582, 6620, 19, 185, 294, 1125, 2777, 4471, 5404, 5417, 5429, 5450, 5828, 5898, 5971, 6147, 6162, 6359, 6414, 6445, 6512, 6589, 6625, 6662, 88, 102, 189, 293, 522, 620, 627, 649, 679, 720, 777, 870, 876, 883, 904, 1234, 1360, 1443, 1497, 1561, 1685, 1751, 1796, 1936, 1991, 2001, 2046, 2050, 2068, 2395, 2514, 2600, 2612, 2628, 2788, 2921, 2941, 2955, 2961, 2969, 3012, 3084, 3164, 3212, 3262, 3327, 3357, 3360, 3396, 3427, 3541, 3546, 3645, 3703, 3733, 3765, 3806, 3894, 4126, 4145, 4190, 4252, 4278, 4460, 4462, 4623, 4659, 4664, 4709, 4765, 4791, 4927, 4952, 4980, 5037, 5100, 5145, 5181, 5224, 5234, 5327, 5434, 5659, 5690, 5786, 5790, 5914, 5972, 5977, 6030, 6338, 6369],
    5: [2177, 3711, 5762, 5950, 6626, 90, 97, 117, 188, 242, 301, 327, 330, 355, 417, 591, 673, 802, 872, 890, 896, 984, 993, 1269, 1318, 1419, 1461, 1518, 1530, 1582, 1743, 1748, 1844, 1907, 1937, 2162, 2212, 2452, 2525, 2642, 2736, 2935, 3115, 3282, 3429, 3510, 3992, 4134, 4493, 4509, 4600, 4616, 4690, 4914, 5153, 5346, 5420, 5775, 5859, 5934, 6006, 6288, 6502, 2195, 2229, 4907, 6510, 143, 251, 676, 681, 717, 865, 922, 1090, 1103, 1157, 1196, 1246, 1266, 1520, 1538, 1745, 1784, 1893, 1950, 2099, 2106, 2220, 2382, 2476, 2534, 2701, 2944, 2947, 3100, 3132, 3161, 3231, 3565, 3753, 3904, 3938, 4050, 4213, 4395, 4495, 4692, 4821, 5039, 5110, 5238, 5427, 5443, 6026, 6256],
    6: [782, 4070, 4382, 5276, 5578, 5703, 5758, 6444, 6555, 6631, 560, 1665, 1728, 1965, 2351, 2605, 2806, 3234, 3452, 3689, 3757, 3852, 4117, 4675, 5235, 5314, 5628, 5715, 5862, 6267, 6380, 6479, 1527, 1825, 1966, 2078, 2742, 3045, 3625, 4773, 4888, 4922, 5374, 5393, 6076, 6368, 6486, 15, 40, 159, 295, 347, 359, 363, 615, 657, 686, 748, 885, 949, 1108, 1188, 1214, 1256, 1283, 1367, 1430, 1545, 1744, 1788, 1793, 1838, 1876, 1903, 2025, 2055, 2076, 2094, 2098, 2196, 2216, 2298, 2305, 2369, 2579, 2741, 2771, 2776, 2889, 2957, 3028, 3040, 3078, 3090, 3224, 3225, 3277, 3328, 3368, 3384, 3448, 3467, 3471, 3478, 3480, 3594, 3681, 3695, 3781, 3857, 3884, 3886, 3901, 4034, 4150, 4151, 4161, 4311, 4313, 4372, 4391, 4400, 4422, 4459, 4504, 4708, 4783, 4811, 4853, 4861, 4873, 5044, 5086, 5202, 5285, 5286, 5478, 5483, 5495, 5499, 5525, 5630, 5974, 6244, 6342, 6384, 6537, 6583],
    7: [292, 645, 707, 867, 893, 954, 1276, 1417, 1499, 1609, 1739, 1795, 1800, 1807, 1880, 1942, 1996, 2130, 2287, 2324, 2533, 2584, 2649, 2704, 2791, 2830, 2845, 2866, 2895, 2906, 3000, 3015, 3042, 3196, 3271, 3303, 3386, 3422, 3464, 3553, 3554, 3572, 3633, 3814, 3881, 3905, 3923, 3977, 4102, 4116, 4370, 4520, 4570, 4854, 4929, 4931, 5036, 5091, 5121, 5148, 5226, 5263, 5498, 5607, 5612, 5685, 5781, 5891, 5967, 6012, 6091, 6113, 6158, 6239, 6321, 6328, 6361, 6490, 6667, 89, 144, 289, 338, 339, 505, 621, 763, 901, 917, 1000, 1016, 1222, 1507, 1528, 1583, 1598, 1646, 1670, 1689, 1873, 1879, 1899, 1939, 2101, 2164, 2214, 2302, 2318, 2331, 2370, 2510, 2560, 2561, 2604, 2615, 2626, 2746, 2779, 2809, 3065, 3174, 3211, 3217, 3241, 3246, 3487, 3785, 4192, 4272, 4279, 4282, 4344, 4360, 4419, 4447, 4776, 4825, 4870, 5061, 5160, 5182, 5230, 5244, 5273, 5442, 5461, 5593, 5606, 5669, 5696, 5746, 5756, 5802, 5856, 5877, 5894, 5968, 6010, 6019, 6073, 6075, 6097, 6125, 6128, 6137, 6156, 6159, 6165, 6197, 6199, 6223, 6241, 6270, 6278, 6301, 6334, 6373, 6386, 6389, 6421, 6436, 6450, 6471, 6481, 6500, 6559, 6606, 6615, 6633, 6666, 64, 200, 220, 291, 426, 458, 559, 576, 789, 909, 915, 921, 936, 953, 1013, 1119, 1162, 1171, 1218, 1244, 1292, 1326, 1375, 1525, 1608, 1613, 1671, 1760, 1860, 1871, 1885, 1932, 2060, 2254, 2349, 2479, 2494, 2509, 2564, 2611, 2623, 2657, 2761, 2892, 3143, 3150, 3266, 3276, 3302, 3369, 3402, 3430, 3492, 3511, 3512, 3591, 3631, 3664, 3706, 3825, 3827, 3837, 3838, 3876, 3962, 3984, 3996, 4007, 4101, 4133, 4141, 4226, 4247, 4255, 4288, 4339, 4350, 4365, 4376, 4379, 4415, 4435, 4512, 4527, 4563, 4598, 4610, 4661, 4681, 4688, 4777, 4961, 4977, 5041, 5106, 5136, 5166, 5252, 5278, 5290, 5310, 5313, 5324, 5329, 5396, 5430, 5431, 5445, 5501, 5532, 5592, 5596, 5661, 5702, 5704, 5834, 5896, 5937, 5985, 6070, 6185, 6212, 6331, 6333, 6433, 6529, 6542, 6566, 6568, 35, 51, 81, 109, 119, 123, 128, 171, 229, 239, 328, 342, 395, 537, 600, 641, 667, 701, 764, 815, 891, 907, 935, 976, 991, 1045, 1051, 1071, 1191, 1223, 1295, 1319, 1377, 1389, 1406, 1458, 1495, 1522, 1529, 1540, 1553, 1710, 1865, 1902, 1922, 2054, 2066, 2073, 2167, 2234, 2292, 2294, 2378, 2392, 2435, 2556, 2586, 2589, 2629, 2640, 2694, 2834, 2847, 2946, 2965, 2999, 3003, 3064, 3095, 3127, 3203, 3237, 3269, 3275, 3319, 3408, 3419, 3486, 3522, 3623, 3688, 3707, 3768, 3771, 3843, 3859, 3863, 3980, 4020, 4038, 4220, 4270, 4381, 4429, 4442, 4456, 4531, 4534, 4545, 4613, 4679, 4685, 4706, 4745, 4758, 4763, 4841, 4953, 4966, 4996, 5064, 5081, 5095, 5099, 5203, 5253, 5323, 5365, 5412, 5425, 5444, 5465, 5565, 5595, 5602, 5682, 5691, 5772, 5847, 5872, 5887, 5935, 6112, 6250, 6381, 6403, 6416, 6508, 6532, 6601, 6639],
    8: [38, 75, 162, 205, 247, 258, 276, 287, 288, 307, 392, 465, 565, 744, 784, 798, 859, 931, 937, 974, 1004, 1077, 1207, 1228, 1331, 1395, 1453, 1490, 1546, 1565, 1607, 1726, 1786, 2020, 2104, 2139, 2168, 2200, 2239, 2371, 2412, 2467, 2487, 2559, 2632, 2635, 2858, 2862, 2934, 2940, 3170, 3477, 3585, 3609, 3832, 3974, 4008, 4059, 4140, 4154, 4264, 4312, 4461, 4490, 4537, 4550, 4667, 4682, 4740, 4782, 4787, 4867, 5029, 5096, 5168, 5170, 5218, 5281, 5293, 5351, 5447, 5456, 5457, 5531, 5641, 5657, 5668, 5726, 5732, 5819, 5926, 5949, 5980, 5991, 6110, 6245, 6289, 6405, 6426, 6474, 6558, 26, 348, 366, 452, 467, 518, 582, 601, 704, 736, 1096, 1164, 1224, 1438, 1541, 1732, 1938, 2063, 2336, 2383, 2459, 2614, 2710, 2798, 2810, 2842, 2918, 3001, 3023, 3353, 3595, 3889, 3981, 4338, 4364, 4408, 4618, 4989, 5177, 5227, 5524, 5566, 5597, 5615, 5622, 5713, 5721, 5725, 5754, 5761, 5794, 5799, 5869, 5886, 5910, 5929, 5942, 5944, 5975, 6001, 6023, 6052, 6068, 6087, 6168, 6191, 6195, 6201, 6218, 6224, 6235, 6246, 6302, 6341, 6366, 6385, 6408, 6425, 6480, 6483, 6485, 6495, 6632, 6634, 8, 24, 140, 211, 224, 235, 250, 337, 356, 402, 438, 443, 454, 489, 496, 530, 566, 583, 592, 659, 675, 716, 760, 812, 832, 848, 853, 868, 902, 941, 960, 978, 1025, 1027, 1058, 1061, 1113, 1173, 1260, 1279, 1298, 1303, 1338, 1341, 1376, 1382, 1388, 1409, 1440, 1491, 1496, 1501, 1506, 1517, 1571, 1572, 1594, 1614, 1621, 1622, 1642, 1652, 1704, 1727, 1801, 1834, 1846, 1886, 1898, 1904, 1911, 1923, 1924, 1967, 1978, 1998, 2031, 2047, 2052, 2087, 2116, 2173, 2213, 2219, 2248, 2257, 2267, 2303, 2319, 2320, 2352, 2353, 2363, 2385, 2407, 2446, 2460, 2501, 2531, 2539, 2542, 2547, 2573, 2578, 2616, 2618, 2665, 2671, 2676, 2690, 2691, 2728, 2739, 2751, 2784, 2786, 2797, 2805, 2840, 2843, 2869, 2911, 2988, 2994, 3071, 3087, 3089, 3171, 3181, 3182, 3186, 3190, 3220, 3274, 3345, 3355, 3364, 3375, 3391, 3425, 3469, 3470, 3479, 3496, 3528, 3536, 3555, 3557, 3562, 3600, 3607, 3661, 3668, 3728, 3762, 3767, 3773, 3840, 3841, 3848, 3868, 3916, 3918, 3919, 3940, 3965, 3969, 3983, 3991, 4121, 4123, 4128, 4153, 4204, 4208, 4250, 4285, 4330, 4331, 4332, 4337, 4352, 4356, 4384, 4492, 4521, 4544, 4549, 4568, 4581, 4588, 4605, 4717, 4727, 4839, 4840, 4850, 4904, 4915, 4945, 4958, 4993, 5004, 5005, 5006, 5065, 5071, 5141, 5151, 5159, 5190, 5213, 5228, 5232, 5294, 5304, 5350, 5403, 5415, 5451, 5511, 5515, 5568, 5613, 5666, 5692, 5701, 5723, 5765, 5791, 5816, 5835, 5890, 5907, 5912, 5966, 6054, 6071, 6078, 6103, 6221, 6263, 6277, 6367, 6379, 6383, 6464, 6497, 6531, 6613, 6641, 6646, 3, 115, 245, 437, 616, 787, 884, 934, 948, 973, 1033, 1189, 1219, 1344, 1408, 1412, 1421, 1442, 1463, 1645, 2015, 2512, 2594, 2652, 2958, 2993, 3146, 3184, 3286, 3866, 3892, 4000, 4079, 4191, 4233, 4485, 4658, 4721, 4741, 4795, 4895, 4970, 5231, 5243, 5262, 5502, 5528, 5548, 5650, 5876, 5959, 5990, 6046, 6063, 6065, 6100, 6131, 6192, 6200, 6260, 6310, 6347, 6475, 6635, 6636],
    9: [153, 472, 543, 727, 830, 873, 1023, 1187, 1212, 1428, 1448, 1457, 1523, 1560, 1673, 1920, 2028, 2189, 2263, 2518, 2575, 2808, 2992, 3051, 3091, 3738, 3741, 4396, 4541, 4666, 4797, 4936, 5092, 5392, 5409, 5506, 5586, 5611, 5663, 5683, 5744, 5768, 5818, 5830, 5924, 6184, 6249, 6271, 6275, 6428, 6477, 6478, 6658, 80, 589, 755, 1324, 1617, 1772, 2687, 2787, 4540, 4807, 5686, 5731, 5749, 5787, 5792, 5838, 5903, 6186, 6227, 6324, 6372, 6394, 6439, 6451, 6520, 6543, 6544, 6560, 6607, 42, 152, 166, 184, 209, 308, 310, 360, 368, 374, 379, 405, 419, 424, 434, 459, 475, 482, 485, 499, 520, 524, 548, 562, 572, 590, 613, 629, 654, 688, 691, 730, 741, 773, 800, 813, 817, 844, 863, 962, 964, 997, 1024, 1026, 1039, 1069, 1088, 1093, 1112, 1131, 1172, 1177, 1225, 1315, 1337, 1358, 1416, 1426, 1431, 1435, 1468, 1493, 1509, 1533, 1559, 1573, 1589, 1640, 1649, 1669, 1703, 1762, 1782, 1810, 1819, 1946, 1993, 2072, 2079, 2089, 2092, 2120, 2129, 2146, 2185, 2194, 2199, 2240, 2250, 2273, 2304, 2315, 2316, 2325, 2330, 2338, 2342, 2408, 2429, 2438, 2439, 2472, 2473, 2520, 2555, 2596, 2603, 2719, 2754, 2760, 2867, 2876, 2899, 2905, 2920, 2928, 2964, 2984, 3022, 3030, 3050, 3053, 3069, 3076, 3080, 3086, 3116, 3118, 3121, 3142, 3172, 3180, 3191, 3215, 3291, 3297, 3318, 3323, 3343, 3374, 3383, 3397, 3458, 3461, 3484, 3509, 3524, 3531, 3592, 3598, 3599, 3603, 3610, 3614, 3619, 3636, 3656, 3671, 3692, 3713, 3760, 3772, 3780, 3796, 3797, 3816, 3821, 3850, 3873, 3877, 3903, 3914, 4024, 4032, 4052, 4068, 4085, 4095, 4110, 4118, 4166, 4168, 4206, 4216, 4224, 4267, 4281, 4290, 4296, 4355, 4406, 4413, 4424, 4425, 4446, 4555, 4591, 4594, 4632, 4650, 4713, 4719, 4729, 4738, 4752, 4761, 4767, 4784, 4788, 4810, 4831, 4856, 4899, 4933, 4960, 4981, 4991, 4995, 5003, 5007, 5016, 5053, 5057, 5067, 5125, 5129, 5223, 5257, 5264, 5269, 5287, 5343, 5347, 5369, 5390, 5395, 5405, 5482, 5509, 5514, 5553, 5560, 5561, 5562, 5618, 5753, 5767, 5796, 5900, 5917, 5933, 5953, 5960, 6007, 6014, 6044, 6064, 6081, 6089, 6099, 6163, 6173, 6181, 6193, 6196, 6298, 6459, 6469, 6546, 6549, 87, 93, 416, 700, 719, 738, 1956, 2048, 2111, 2516, 2883, 3399, 4389, 4848, 4849, 5132, 5282, 5460, 5464, 5471, 5535, 5569, 5705, 5969, 6190, 6255, 6343, 6476],
    10: [5, 124, 361, 406, 453, 642, 986, 1018, 1030, 1086, 1107, 1115, 1178, 1471, 1536, 1718, 1780, 1852, 1863, 1984, 2148, 2209, 2295, 2484, 2490, 2515, 2703, 2727, 2799, 2937, 2939, 3059, 3160, 3173, 3222, 3265, 3296, 3340, 3352, 3543, 3569, 3574, 3576, 3618, 3626, 3628, 3676, 3743, 3802, 3817, 3958, 3979, 4179, 4293, 4320, 4463, 4532, 4656, 4662, 4687, 4864, 4911, 5056, 5101, 5186, 5196, 5260, 5322, 5739, 5771, 5797, 5885, 6025, 6069, 6108, 6174, 6219, 6258, 6420, 6545, 1296, 1332, 1663, 1847, 2057, 2233, 2258, 2680, 2718, 3041, 3601, 3677, 4407, 4430, 4597, 4836, 4898, 4951, 5256, 5401, 5573, 5648, 5653, 5717, 5783, 6045, 6129, 6141, 6176, 6262, 6303, 6449, 6482, 6629, 6, 29, 49, 131, 133, 157, 183, 216, 234, 246, 249, 254, 286, 305, 319, 329, 351, 373, 391, 422, 429, 466, 477, 500, 504, 507, 618, 635, 647, 650, 662, 685, 718, 737, 739, 751, 757, 759, 785, 837, 842, 852, 862, 910, 955, 989, 992, 1052, 1059, 1095, 1114, 1190, 1259, 1270, 1272, 1320, 1370, 1404, 1407, 1427, 1449, 1544, 1547, 1584, 1627, 1641, 1653, 1696, 1736, 1804, 1835, 1891, 1892, 1940, 2018, 2085, 2124, 2145, 2154, 2156, 2184, 2227, 2246, 2332, 2343, 2358, 2419, 2482, 2517, 2529, 2537, 2549, 2557, 2580, 2647, 2666, 2683, 2744, 2747, 2803, 2833, 2836, 2841, 2891, 2898, 2936, 2954, 2962, 2980, 3005, 3010, 3013, 3017, 3019, 3032, 3052, 3123, 3129, 3157, 3178, 3201, 3239, 3256, 3310, 3316, 3362, 3371, 3404, 3433, 3435, 3474, 3503, 3517, 3545, 3638, 3678, 3731, 3790, 3793, 3800, 3824, 3828, 3851, 3912, 3952, 4071, 4082, 4111, 4130, 4144, 4225, 4242, 4243, 4276, 4283, 4291, 4297, 4300, 4358, 4361, 4436, 4438, 4439, 4511, 4523, 4571, 4578, 4614, 4624, 4634, 4637, 4642, 4644, 4651, 4672, 4698, 4753, 4771, 4815, 4834, 4937, 5066, 5130, 5163, 5207, 5215, 5248, 5255, 5261, 5326, 5330, 5381, 5384, 5435, 5437, 5453, 5474, 5538, 5555, 5660, 5722, 5784, 5827, 5952, 6188, 6215, 6453, 6484, 1200, 1921, 1972, 2597, 2677, 2756, 2900, 3156, 3749, 3810, 4091, 4135, 4222, 4617, 4754, 4829, 4938, 5332, 5491, 5646, 5947, 6036, 6253, 6291, 6317, 6377, 6434, 6612],
    11: [9, 37, 69, 82, 113, 149, 163, 203, 261, 262, 263, 264, 390, 394, 430, 478, 488, 567, 577, 581, 606, 622, 631, 638, 672, 769, 772, 833, 1001, 1134, 1165, 1238, 1316, 1379, 1387, 1511, 1534, 1542, 1556, 1591, 1620, 1706, 1722, 1763, 1774, 1776, 1849, 1867, 1874, 1878, 1925, 1928, 1941, 1954, 1955, 1968, 2014, 2064, 2107, 2109, 2179, 2187, 2224, 2309, 2354, 2396, 2424, 2466, 2483, 2553, 2630, 2660, 2699, 2702, 2737, 2738, 2794, 2817, 2828, 2837, 2879, 2970, 2997, 3029, 3058, 3085, 3207, 3228, 3238, 3356, 3381, 3394, 3409, 3417, 3456, 3488, 3532, 3634, 3672, 3748, 3759, 3764, 3833, 3854, 3874, 3899, 3907, 3908, 3966, 4029, 4043, 4092, 4254, 4305, 4405, 4411, 4431, 4476, 4535, 4645, 4671, 4704, 4731, 4766, 4790, 4804, 4852, 4909, 4969, 5087, 5124, 5167, 5173, 5174, 5188, 5216, 5272, 5335, 5364, 5487, 5537, 5634, 5658, 5736, 5803, 5918, 6077, 6090, 6124, 6205, 6247, 6365, 6651, 122, 866, 925, 1037, 1078, 1118, 1123, 1855, 2095, 2386, 2409, 2631, 2880, 3794, 3799, 4149, 4530, 4649, 4746, 4905, 4983, 5021, 5183, 5249, 5433, 5598, 5740, 5871, 5963, 6042, 6048, 6094, 6143, 6208, 6305, 6382, 6398, 6446, 6457, 6551, 6577, 6609, 6622, 43, 215, 217, 225, 268, 298, 375, 410, 425, 457, 492, 506, 514, 564, 573, 585, 608, 626, 776, 788, 899, 933, 1009, 1068, 1127, 1251, 1265, 1305, 1414, 1420, 1452, 1462, 1479, 1539, 1602, 1976, 2019, 2027, 2036, 2084, 2314, 2356, 2442, 2444, 2491, 2554, 2619, 2663, 2686, 2878, 3020, 3073, 3120, 3131, 3149, 3270, 3279, 3295, 3301, 3367, 3405, 3434, 3450, 3653, 3698, 3730, 3805, 3847, 4097, 4125, 4194, 4292, 4448, 4454, 4487, 4488, 4823, 4925, 4965, 4968, 5301, 5387, 5422, 5631, 5698, 5745, 5764, 5911, 6033, 6057, 6161, 6206, 6282, 6374, 6393, 6417, 6547, 6616, 33, 56, 79, 112, 194, 299, 617, 623, 692, 816, 924, 938, 939, 1007, 1046, 1049, 1169, 1323, 1469, 1636, 1681, 2102, 2208, 2244, 2245, 2296, 2361, 2470, 2929, 2973, 3060, 3122, 3219, 3285, 3329, 3336, 3347, 3468, 3680, 3694, 3732, 4075, 4335, 4345, 4386, 4824, 5023, 5277, 5361, 5492, 5552, 5627, 5687, 5688, 6002, 6040, 6150, 6166, 6187, 6268, 6294, 6454, 6524],
    12: [20, 21, 57, 107, 129, 150, 168, 214, 218, 265, 420, 451, 517, 546, 578, 665, 682, 696, 1011, 1215, 1216, 1267, 1632, 1656, 1693, 1824, 1831, 1929, 1969, 1974, 1983, 1987, 2029, 2069, 2181, 2191, 2210, 2228, 2264, 2289, 2390, 2423, 2433, 2480, 2508, 2613, 2643, 2807, 2874, 2989, 3054, 3061, 3106, 3218, 3258, 3393, 3457, 3505, 3518, 3534, 3690, 3788, 3861, 4042, 4104, 4147, 4203, 4260, 4287, 4302, 4440, 4458, 4483, 4602, 4635, 4648, 4683, 4774, 4803, 4845, 4891, 4932, 5045, 5375, 5604, 5645, 5689, 5776, 5777, 5922, 5936, 6041, 6106, 6170, 6300, 6350, 6534, 487, 683, 790, 838, 1003, 1014, 1031, 1097, 1137, 1151, 1273, 1402, 1592, 1629, 1668, 1674, 1737, 1858, 1877, 1883, 1930, 1960, 1992, 2038, 2075, 2155, 2241, 2251, 2262, 2367, 2441, 2593, 2608, 2846, 2924, 2956, 3006, 3018, 3025, 3104, 3124, 3162, 3198, 3247, 3272, 3321, 3322, 3354, 3442, 3539, 3567, 3579, 3742, 3761, 3831, 3858, 3906, 3987, 4099, 4124, 4158, 4162, 4261, 4449, 4468, 4478, 4501, 4518, 4552, 4585, 4590, 4615, 4646, 4714, 4726, 4733, 4736, 4748, 4762, 4778, 4819, 4830, 4866, 4868, 4884, 4916, 4974, 4990, 5002, 5008, 5011, 5089, 5126, 5158, 5211, 5318, 5410, 5414, 5419, 5424, 5455, 5459, 5534, 5580, 5620, 5667, 5671, 5677, 5724, 5748, 5774, 5851, 5854, 5945, 5988, 6009, 6011, 6029, 6047, 6049, 6083, 6093, 6122, 6154, 6222, 6351, 6355, 6517, 6573, 6624, 6671, 161, 332, 403, 494, 534, 677, 803, 839, 855, 871, 908, 1122, 1150, 1195, 1236, 1396, 1456, 1734, 2000, 2462, 2609, 2755, 2923, 3014, 3046, 3133, 3147, 3158, 3229, 3313, 3349, 3648, 3872, 3971, 4019, 4081, 4198, 4236, 4374, 4543, 4816, 5019, 5134, 5550, 5617, 5624, 5680, 5741, 5766, 5915, 5992, 6062, 6088, 6148, 6204, 6234, 6242, 6313, 6322, 6329, 6437, 6505, 6554, 6611, 48, 50, 105, 120, 444, 470, 474, 484, 644, 979, 1080, 1226, 1233, 1257, 1294, 1302, 1330, 1418, 1657, 1658, 1662, 1715, 1716, 1857, 1945, 1986, 2004, 2153, 2193, 2226, 2402, 2443, 2634, 2648, 2697, 2764, 2781, 2796, 2860, 2907, 3048, 3055, 3074, 3098, 3103, 3105, 3112, 3183, 3192, 3200, 3299, 3501, 3551, 3561, 3621, 3641, 3658, 3704, 3705, 3745, 3795, 3887, 3900, 3935, 3937, 3976, 4041, 4143, 4189, 4200, 4207, 4295, 4377, 4434, 4444, 4481, 4507, 4607, 4627, 4744, 4878, 4892, 4900, 4901, 4950, 5030, 5112, 5123, 5250, 5306, 5309, 5348, 5376, 5503, 5520, 5601, 5629, 5759, 5905, 5908, 6016, 6114, 6276, 6378, 6388, 6438, 6456, 6487, 6516, 6518, 6525]
}