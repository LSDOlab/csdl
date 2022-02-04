"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[2103,4787],{3905:function(e,n,t){t.d(n,{Zo:function(){return m},kt:function(){return d}});var r=t(7294);function o(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function s(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function a(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?s(Object(t),!0).forEach((function(n){o(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):s(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function l(e,n){if(null==e)return{};var t,r,o=function(e,n){if(null==e)return{};var t,r,o={},s=Object.keys(e);for(r=0;r<s.length;r++)t=s[r],n.indexOf(t)>=0||(o[t]=e[t]);return o}(e,n);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(r=0;r<s.length;r++)t=s[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}var i=r.createContext({}),u=function(e){var n=r.useContext(i),t=n;return e&&(t="function"==typeof e?e(n):a(a({},n),e)),t},m=function(e){var n=u(e.components);return r.createElement(i.Provider,{value:n},e.children)},p={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},c=r.forwardRef((function(e,n){var t=e.components,o=e.mdxType,s=e.originalType,i=e.parentName,m=l(e,["components","mdxType","originalType","parentName"]),c=u(t),d=o,f=c["".concat(i,".").concat(d)]||c[d]||p[d]||s;return t?r.createElement(f,a(a({ref:n},m),{},{components:t})):r.createElement(f,a({ref:n},m))}));function d(e,n){var t=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var s=t.length,a=new Array(s);a[0]=c;var l={};for(var i in n)hasOwnProperty.call(n,i)&&(l[i]=n[i]);l.originalType=e,l.mdxType="string"==typeof e?e:o,a[1]=l;for(var u=2;u<s;u++)a[u]=t[u];return r.createElement.apply(null,a)}return r.createElement.apply(null,t)}c.displayName="MDXCreateElement"},2922:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return i},contentTitle:function(){return u},metadata:function(){return m},toc:function(){return p},default:function(){return d}});var r=t(7462),o=t(3366),s=(t(7294),t(3905)),a=t(4391),l=["components"],i={},u="Sum of Multiple Tensors",m={unversionedId:"examples/Standard Library/sum/ex_sum_multiple_tensor",id:"examples/Standard Library/sum/ex_sum_multiple_tensor",isDocsHomePage:!1,title:"Sum of Multiple Tensors",description:"This is an example of computing the sum of a multiple tensor inputs.",source:"@site/docs/examples/Standard Library/sum/ex_sum_multiple_tensor.mdx",sourceDirName:"examples/Standard Library/sum",slug:"/examples/Standard Library/sum/ex_sum_multiple_tensor",permalink:"/csdl/docs/examples/Standard Library/sum/ex_sum_multiple_tensor",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/sum/ex_sum_multiple_tensor.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Sum of Multiple Matrices along Rows",permalink:"/csdl/docs/examples/Standard Library/sum/ex_sum_multiple_matrix_along1"},next:{title:"Sum of Multiple Vectors",permalink:"/csdl/docs/examples/Standard Library/sum/ex_sum_multiple_vector"}},p=[],c={toc:p};function d(e){var n=e.components,t=(0,o.Z)(e,l);return(0,s.kt)("wrapper",(0,r.Z)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,s.kt)("h1",{id:"sum-of-multiple-tensors"},"Sum of Multiple Tensors"),(0,s.kt)("p",null,"This is an example of computing the sum of a multiple tensor inputs."),(0,s.kt)(a.default,{mdxType:"WorkedExample"}))}d.isMDXComponent=!0},4391:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return l},contentTitle:function(){return i},metadata:function(){return u},toc:function(){return m},default:function(){return c}});var r=t(7462),o=t(3366),s=(t(7294),t(3905)),a=["components"],l={},i=void 0,u={unversionedId:"worked_examples/ex_sum_multiple_tensor",id:"worked_examples/ex_sum_multiple_tensor",isDocsHomePage:!1,title:"ex_sum_multiple_tensor",description:"`py",source:"@site/docs/worked_examples/ex_sum_multiple_tensor.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_sum_multiple_tensor",permalink:"/csdl/docs/worked_examples/ex_sum_multiple_tensor",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_sum_multiple_tensor.mdx",tags:[],version:"current",frontMatter:{}},m=[],p={toc:m};function c(e){var n=e.components,t=(0,o.Z)(e,a);return(0,s.kt)("wrapper",(0,r.Z)({},p,t,{components:n,mdxType:"MDXLayout"}),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleMultipleTensor(Model):\n\n    def define(self):\n        n = 3\n        m = 6\n        p = 7\n        q = 10\n\n        # Declare a tensor of shape 3x6x7x10 as input\n        T1 = self.declare_variable('T1',\n                                   val=np.arange(n * m * p * q).reshape(\n                                       (n, m, p, q)))\n\n        # Declare another tensor of shape 3x6x7x10 as input\n        T2 = self.declare_variable('T2',\n                                   val=np.arange(n * m * p * q, 2 * n *\n                                                 m * p * q).reshape(\n                                                     (n, m, p, q)))\n\n        # Output the elementwise sum of tensors T1 and T2\n        self.register_output('multiple_tensor_sum', csdl.sum(T1, T2))\n\n\nsim = Simulator(ExampleMultipleTensor())\nsim.run()\n\nprint('T1', sim['T1'].shape)\nprint(sim['T1'])\nprint('T2', sim['T2'].shape)\nprint(sim['T2'])\nprint('multiple_tensor_sum', sim['multiple_tensor_sum'].shape)\nprint(sim['multiple_tensor_sum'])\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-T1",metastring:"(3, 6, 7, 10)","(3,":!0,"6,":!0,"7,":!0,"10)":!0},"[[[[0.000e+00 1.000e+00 2.000e+00 ... 7.000e+00 8.000e+00 9.000e+00]\n   [1.000e+01 1.100e+01 1.200e+01 ... 1.700e+01 1.800e+01 1.900e+01]\n   [2.000e+01 2.100e+01 2.200e+01 ... 2.700e+01 2.800e+01 2.900e+01]\n   ...\n   [4.000e+01 4.100e+01 4.200e+01 ... 4.700e+01 4.800e+01 4.900e+01]\n   [5.000e+01 5.100e+01 5.200e+01 ... 5.700e+01 5.800e+01 5.900e+01]\n   [6.000e+01 6.100e+01 6.200e+01 ... 6.700e+01 6.800e+01 6.900e+01]]\n\n  [[7.000e+01 7.100e+01 7.200e+01 ... 7.700e+01 7.800e+01 7.900e+01]\n   [8.000e+01 8.100e+01 8.200e+01 ... 8.700e+01 8.800e+01 8.900e+01]\n   [9.000e+01 9.100e+01 9.200e+01 ... 9.700e+01 9.800e+01 9.900e+01]\n   ...\n   [1.100e+02 1.110e+02 1.120e+02 ... 1.170e+02 1.180e+02 1.190e+02]\n   [1.200e+02 1.210e+02 1.220e+02 ... 1.270e+02 1.280e+02 1.290e+02]\n   [1.300e+02 1.310e+02 1.320e+02 ... 1.370e+02 1.380e+02 1.390e+02]]\n\n  [[1.400e+02 1.410e+02 1.420e+02 ... 1.470e+02 1.480e+02 1.490e+02]\n   [1.500e+02 1.510e+02 1.520e+02 ... 1.570e+02 1.580e+02 1.590e+02]\n   [1.600e+02 1.610e+02 1.620e+02 ... 1.670e+02 1.680e+02 1.690e+02]\n   ...\n   [1.800e+02 1.810e+02 1.820e+02 ... 1.870e+02 1.880e+02 1.890e+02]\n   [1.900e+02 1.910e+02 1.920e+02 ... 1.970e+02 1.980e+02 1.990e+02]\n   [2.000e+02 2.010e+02 2.020e+02 ... 2.070e+02 2.080e+02 2.090e+02]]\n\n  [[2.100e+02 2.110e+02 2.120e+02 ... 2.170e+02 2.180e+02 2.190e+02]\n   [2.200e+02 2.210e+02 2.220e+02 ... 2.270e+02 2.280e+02 2.290e+02]\n   [2.300e+02 2.310e+02 2.320e+02 ... 2.370e+02 2.380e+02 2.390e+02]\n   ...\n   [2.500e+02 2.510e+02 2.520e+02 ... 2.570e+02 2.580e+02 2.590e+02]\n   [2.600e+02 2.610e+02 2.620e+02 ... 2.670e+02 2.680e+02 2.690e+02]\n   [2.700e+02 2.710e+02 2.720e+02 ... 2.770e+02 2.780e+02 2.790e+02]]\n\n  [[2.800e+02 2.810e+02 2.820e+02 ... 2.870e+02 2.880e+02 2.890e+02]\n   [2.900e+02 2.910e+02 2.920e+02 ... 2.970e+02 2.980e+02 2.990e+02]\n   [3.000e+02 3.010e+02 3.020e+02 ... 3.070e+02 3.080e+02 3.090e+02]\n   ...\n   [3.200e+02 3.210e+02 3.220e+02 ... 3.270e+02 3.280e+02 3.290e+02]\n   [3.300e+02 3.310e+02 3.320e+02 ... 3.370e+02 3.380e+02 3.390e+02]\n   [3.400e+02 3.410e+02 3.420e+02 ... 3.470e+02 3.480e+02 3.490e+02]]\n\n  [[3.500e+02 3.510e+02 3.520e+02 ... 3.570e+02 3.580e+02 3.590e+02]\n   [3.600e+02 3.610e+02 3.620e+02 ... 3.670e+02 3.680e+02 3.690e+02]\n   [3.700e+02 3.710e+02 3.720e+02 ... 3.770e+02 3.780e+02 3.790e+02]\n   ...\n   [3.900e+02 3.910e+02 3.920e+02 ... 3.970e+02 3.980e+02 3.990e+02]\n   [4.000e+02 4.010e+02 4.020e+02 ... 4.070e+02 4.080e+02 4.090e+02]\n   [4.100e+02 4.110e+02 4.120e+02 ... 4.170e+02 4.180e+02 4.190e+02]]]\n\n\n [[[4.200e+02 4.210e+02 4.220e+02 ... 4.270e+02 4.280e+02 4.290e+02]\n   [4.300e+02 4.310e+02 4.320e+02 ... 4.370e+02 4.380e+02 4.390e+02]\n   [4.400e+02 4.410e+02 4.420e+02 ... 4.470e+02 4.480e+02 4.490e+02]\n   ...\n   [4.600e+02 4.610e+02 4.620e+02 ... 4.670e+02 4.680e+02 4.690e+02]\n   [4.700e+02 4.710e+02 4.720e+02 ... 4.770e+02 4.780e+02 4.790e+02]\n   [4.800e+02 4.810e+02 4.820e+02 ... 4.870e+02 4.880e+02 4.890e+02]]\n\n  [[4.900e+02 4.910e+02 4.920e+02 ... 4.970e+02 4.980e+02 4.990e+02]\n   [5.000e+02 5.010e+02 5.020e+02 ... 5.070e+02 5.080e+02 5.090e+02]\n   [5.100e+02 5.110e+02 5.120e+02 ... 5.170e+02 5.180e+02 5.190e+02]\n   ...\n   [5.300e+02 5.310e+02 5.320e+02 ... 5.370e+02 5.380e+02 5.390e+02]\n   [5.400e+02 5.410e+02 5.420e+02 ... 5.470e+02 5.480e+02 5.490e+02]\n   [5.500e+02 5.510e+02 5.520e+02 ... 5.570e+02 5.580e+02 5.590e+02]]\n\n  [[5.600e+02 5.610e+02 5.620e+02 ... 5.670e+02 5.680e+02 5.690e+02]\n   [5.700e+02 5.710e+02 5.720e+02 ... 5.770e+02 5.780e+02 5.790e+02]\n   [5.800e+02 5.810e+02 5.820e+02 ... 5.870e+02 5.880e+02 5.890e+02]\n   ...\n   [6.000e+02 6.010e+02 6.020e+02 ... 6.070e+02 6.080e+02 6.090e+02]\n   [6.100e+02 6.110e+02 6.120e+02 ... 6.170e+02 6.180e+02 6.190e+02]\n   [6.200e+02 6.210e+02 6.220e+02 ... 6.270e+02 6.280e+02 6.290e+02]]\n\n  [[6.300e+02 6.310e+02 6.320e+02 ... 6.370e+02 6.380e+02 6.390e+02]\n   [6.400e+02 6.410e+02 6.420e+02 ... 6.470e+02 6.480e+02 6.490e+02]\n   [6.500e+02 6.510e+02 6.520e+02 ... 6.570e+02 6.580e+02 6.590e+02]\n   ...\n   [6.700e+02 6.710e+02 6.720e+02 ... 6.770e+02 6.780e+02 6.790e+02]\n   [6.800e+02 6.810e+02 6.820e+02 ... 6.870e+02 6.880e+02 6.890e+02]\n   [6.900e+02 6.910e+02 6.920e+02 ... 6.970e+02 6.980e+02 6.990e+02]]\n\n  [[7.000e+02 7.010e+02 7.020e+02 ... 7.070e+02 7.080e+02 7.090e+02]\n   [7.100e+02 7.110e+02 7.120e+02 ... 7.170e+02 7.180e+02 7.190e+02]\n   [7.200e+02 7.210e+02 7.220e+02 ... 7.270e+02 7.280e+02 7.290e+02]\n   ...\n   [7.400e+02 7.410e+02 7.420e+02 ... 7.470e+02 7.480e+02 7.490e+02]\n   [7.500e+02 7.510e+02 7.520e+02 ... 7.570e+02 7.580e+02 7.590e+02]\n   [7.600e+02 7.610e+02 7.620e+02 ... 7.670e+02 7.680e+02 7.690e+02]]\n\n  [[7.700e+02 7.710e+02 7.720e+02 ... 7.770e+02 7.780e+02 7.790e+02]\n   [7.800e+02 7.810e+02 7.820e+02 ... 7.870e+02 7.880e+02 7.890e+02]\n   [7.900e+02 7.910e+02 7.920e+02 ... 7.970e+02 7.980e+02 7.990e+02]\n   ...\n   [8.100e+02 8.110e+02 8.120e+02 ... 8.170e+02 8.180e+02 8.190e+02]\n   [8.200e+02 8.210e+02 8.220e+02 ... 8.270e+02 8.280e+02 8.290e+02]\n   [8.300e+02 8.310e+02 8.320e+02 ... 8.370e+02 8.380e+02 8.390e+02]]]\n\n\n [[[8.400e+02 8.410e+02 8.420e+02 ... 8.470e+02 8.480e+02 8.490e+02]\n   [8.500e+02 8.510e+02 8.520e+02 ... 8.570e+02 8.580e+02 8.590e+02]\n   [8.600e+02 8.610e+02 8.620e+02 ... 8.670e+02 8.680e+02 8.690e+02]\n   ...\n   [8.800e+02 8.810e+02 8.820e+02 ... 8.870e+02 8.880e+02 8.890e+02]\n   [8.900e+02 8.910e+02 8.920e+02 ... 8.970e+02 8.980e+02 8.990e+02]\n   [9.000e+02 9.010e+02 9.020e+02 ... 9.070e+02 9.080e+02 9.090e+02]]\n\n  [[9.100e+02 9.110e+02 9.120e+02 ... 9.170e+02 9.180e+02 9.190e+02]\n   [9.200e+02 9.210e+02 9.220e+02 ... 9.270e+02 9.280e+02 9.290e+02]\n   [9.300e+02 9.310e+02 9.320e+02 ... 9.370e+02 9.380e+02 9.390e+02]\n   ...\n   [9.500e+02 9.510e+02 9.520e+02 ... 9.570e+02 9.580e+02 9.590e+02]\n   [9.600e+02 9.610e+02 9.620e+02 ... 9.670e+02 9.680e+02 9.690e+02]\n   [9.700e+02 9.710e+02 9.720e+02 ... 9.770e+02 9.780e+02 9.790e+02]]\n\n  [[9.800e+02 9.810e+02 9.820e+02 ... 9.870e+02 9.880e+02 9.890e+02]\n   [9.900e+02 9.910e+02 9.920e+02 ... 9.970e+02 9.980e+02 9.990e+02]\n   [1.000e+03 1.001e+03 1.002e+03 ... 1.007e+03 1.008e+03 1.009e+03]\n   ...\n   [1.020e+03 1.021e+03 1.022e+03 ... 1.027e+03 1.028e+03 1.029e+03]\n   [1.030e+03 1.031e+03 1.032e+03 ... 1.037e+03 1.038e+03 1.039e+03]\n   [1.040e+03 1.041e+03 1.042e+03 ... 1.047e+03 1.048e+03 1.049e+03]]\n\n  [[1.050e+03 1.051e+03 1.052e+03 ... 1.057e+03 1.058e+03 1.059e+03]\n   [1.060e+03 1.061e+03 1.062e+03 ... 1.067e+03 1.068e+03 1.069e+03]\n   [1.070e+03 1.071e+03 1.072e+03 ... 1.077e+03 1.078e+03 1.079e+03]\n   ...\n   [1.090e+03 1.091e+03 1.092e+03 ... 1.097e+03 1.098e+03 1.099e+03]\n   [1.100e+03 1.101e+03 1.102e+03 ... 1.107e+03 1.108e+03 1.109e+03]\n   [1.110e+03 1.111e+03 1.112e+03 ... 1.117e+03 1.118e+03 1.119e+03]]\n\n  [[1.120e+03 1.121e+03 1.122e+03 ... 1.127e+03 1.128e+03 1.129e+03]\n   [1.130e+03 1.131e+03 1.132e+03 ... 1.137e+03 1.138e+03 1.139e+03]\n   [1.140e+03 1.141e+03 1.142e+03 ... 1.147e+03 1.148e+03 1.149e+03]\n   ...\n   [1.160e+03 1.161e+03 1.162e+03 ... 1.167e+03 1.168e+03 1.169e+03]\n   [1.170e+03 1.171e+03 1.172e+03 ... 1.177e+03 1.178e+03 1.179e+03]\n   [1.180e+03 1.181e+03 1.182e+03 ... 1.187e+03 1.188e+03 1.189e+03]]\n\n  [[1.190e+03 1.191e+03 1.192e+03 ... 1.197e+03 1.198e+03 1.199e+03]\n   [1.200e+03 1.201e+03 1.202e+03 ... 1.207e+03 1.208e+03 1.209e+03]\n   [1.210e+03 1.211e+03 1.212e+03 ... 1.217e+03 1.218e+03 1.219e+03]\n   ...\n   [1.230e+03 1.231e+03 1.232e+03 ... 1.237e+03 1.238e+03 1.239e+03]\n   [1.240e+03 1.241e+03 1.242e+03 ... 1.247e+03 1.248e+03 1.249e+03]\n   [1.250e+03 1.251e+03 1.252e+03 ... 1.257e+03 1.258e+03 1.259e+03]]]]\nT2 (3, 6, 7, 10)\n[[[[1260. 1261. 1262. ... 1267. 1268. 1269.]\n   [1270. 1271. 1272. ... 1277. 1278. 1279.]\n   [1280. 1281. 1282. ... 1287. 1288. 1289.]\n   ...\n   [1300. 1301. 1302. ... 1307. 1308. 1309.]\n   [1310. 1311. 1312. ... 1317. 1318. 1319.]\n   [1320. 1321. 1322. ... 1327. 1328. 1329.]]\n\n  [[1330. 1331. 1332. ... 1337. 1338. 1339.]\n   [1340. 1341. 1342. ... 1347. 1348. 1349.]\n   [1350. 1351. 1352. ... 1357. 1358. 1359.]\n   ...\n   [1370. 1371. 1372. ... 1377. 1378. 1379.]\n   [1380. 1381. 1382. ... 1387. 1388. 1389.]\n   [1390. 1391. 1392. ... 1397. 1398. 1399.]]\n\n  [[1400. 1401. 1402. ... 1407. 1408. 1409.]\n   [1410. 1411. 1412. ... 1417. 1418. 1419.]\n   [1420. 1421. 1422. ... 1427. 1428. 1429.]\n   ...\n   [1440. 1441. 1442. ... 1447. 1448. 1449.]\n   [1450. 1451. 1452. ... 1457. 1458. 1459.]\n   [1460. 1461. 1462. ... 1467. 1468. 1469.]]\n\n  [[1470. 1471. 1472. ... 1477. 1478. 1479.]\n   [1480. 1481. 1482. ... 1487. 1488. 1489.]\n   [1490. 1491. 1492. ... 1497. 1498. 1499.]\n   ...\n   [1510. 1511. 1512. ... 1517. 1518. 1519.]\n   [1520. 1521. 1522. ... 1527. 1528. 1529.]\n   [1530. 1531. 1532. ... 1537. 1538. 1539.]]\n\n  [[1540. 1541. 1542. ... 1547. 1548. 1549.]\n   [1550. 1551. 1552. ... 1557. 1558. 1559.]\n   [1560. 1561. 1562. ... 1567. 1568. 1569.]\n   ...\n   [1580. 1581. 1582. ... 1587. 1588. 1589.]\n   [1590. 1591. 1592. ... 1597. 1598. 1599.]\n   [1600. 1601. 1602. ... 1607. 1608. 1609.]]\n\n  [[1610. 1611. 1612. ... 1617. 1618. 1619.]\n   [1620. 1621. 1622. ... 1627. 1628. 1629.]\n   [1630. 1631. 1632. ... 1637. 1638. 1639.]\n   ...\n   [1650. 1651. 1652. ... 1657. 1658. 1659.]\n   [1660. 1661. 1662. ... 1667. 1668. 1669.]\n   [1670. 1671. 1672. ... 1677. 1678. 1679.]]]\n\n\n [[[1680. 1681. 1682. ... 1687. 1688. 1689.]\n   [1690. 1691. 1692. ... 1697. 1698. 1699.]\n   [1700. 1701. 1702. ... 1707. 1708. 1709.]\n   ...\n   [1720. 1721. 1722. ... 1727. 1728. 1729.]\n   [1730. 1731. 1732. ... 1737. 1738. 1739.]\n   [1740. 1741. 1742. ... 1747. 1748. 1749.]]\n\n  [[1750. 1751. 1752. ... 1757. 1758. 1759.]\n   [1760. 1761. 1762. ... 1767. 1768. 1769.]\n   [1770. 1771. 1772. ... 1777. 1778. 1779.]\n   ...\n   [1790. 1791. 1792. ... 1797. 1798. 1799.]\n   [1800. 1801. 1802. ... 1807. 1808. 1809.]\n   [1810. 1811. 1812. ... 1817. 1818. 1819.]]\n\n  [[1820. 1821. 1822. ... 1827. 1828. 1829.]\n   [1830. 1831. 1832. ... 1837. 1838. 1839.]\n   [1840. 1841. 1842. ... 1847. 1848. 1849.]\n   ...\n   [1860. 1861. 1862. ... 1867. 1868. 1869.]\n   [1870. 1871. 1872. ... 1877. 1878. 1879.]\n   [1880. 1881. 1882. ... 1887. 1888. 1889.]]\n\n  [[1890. 1891. 1892. ... 1897. 1898. 1899.]\n   [1900. 1901. 1902. ... 1907. 1908. 1909.]\n   [1910. 1911. 1912. ... 1917. 1918. 1919.]\n   ...\n   [1930. 1931. 1932. ... 1937. 1938. 1939.]\n   [1940. 1941. 1942. ... 1947. 1948. 1949.]\n   [1950. 1951. 1952. ... 1957. 1958. 1959.]]\n\n  [[1960. 1961. 1962. ... 1967. 1968. 1969.]\n   [1970. 1971. 1972. ... 1977. 1978. 1979.]\n   [1980. 1981. 1982. ... 1987. 1988. 1989.]\n   ...\n   [2000. 2001. 2002. ... 2007. 2008. 2009.]\n   [2010. 2011. 2012. ... 2017. 2018. 2019.]\n   [2020. 2021. 2022. ... 2027. 2028. 2029.]]\n\n  [[2030. 2031. 2032. ... 2037. 2038. 2039.]\n   [2040. 2041. 2042. ... 2047. 2048. 2049.]\n   [2050. 2051. 2052. ... 2057. 2058. 2059.]\n   ...\n   [2070. 2071. 2072. ... 2077. 2078. 2079.]\n   [2080. 2081. 2082. ... 2087. 2088. 2089.]\n   [2090. 2091. 2092. ... 2097. 2098. 2099.]]]\n\n\n [[[2100. 2101. 2102. ... 2107. 2108. 2109.]\n   [2110. 2111. 2112. ... 2117. 2118. 2119.]\n   [2120. 2121. 2122. ... 2127. 2128. 2129.]\n   ...\n   [2140. 2141. 2142. ... 2147. 2148. 2149.]\n   [2150. 2151. 2152. ... 2157. 2158. 2159.]\n   [2160. 2161. 2162. ... 2167. 2168. 2169.]]\n\n  [[2170. 2171. 2172. ... 2177. 2178. 2179.]\n   [2180. 2181. 2182. ... 2187. 2188. 2189.]\n   [2190. 2191. 2192. ... 2197. 2198. 2199.]\n   ...\n   [2210. 2211. 2212. ... 2217. 2218. 2219.]\n   [2220. 2221. 2222. ... 2227. 2228. 2229.]\n   [2230. 2231. 2232. ... 2237. 2238. 2239.]]\n\n  [[2240. 2241. 2242. ... 2247. 2248. 2249.]\n   [2250. 2251. 2252. ... 2257. 2258. 2259.]\n   [2260. 2261. 2262. ... 2267. 2268. 2269.]\n   ...\n   [2280. 2281. 2282. ... 2287. 2288. 2289.]\n   [2290. 2291. 2292. ... 2297. 2298. 2299.]\n   [2300. 2301. 2302. ... 2307. 2308. 2309.]]\n\n  [[2310. 2311. 2312. ... 2317. 2318. 2319.]\n   [2320. 2321. 2322. ... 2327. 2328. 2329.]\n   [2330. 2331. 2332. ... 2337. 2338. 2339.]\n   ...\n   [2350. 2351. 2352. ... 2357. 2358. 2359.]\n   [2360. 2361. 2362. ... 2367. 2368. 2369.]\n   [2370. 2371. 2372. ... 2377. 2378. 2379.]]\n\n  [[2380. 2381. 2382. ... 2387. 2388. 2389.]\n   [2390. 2391. 2392. ... 2397. 2398. 2399.]\n   [2400. 2401. 2402. ... 2407. 2408. 2409.]\n   ...\n   [2420. 2421. 2422. ... 2427. 2428. 2429.]\n   [2430. 2431. 2432. ... 2437. 2438. 2439.]\n   [2440. 2441. 2442. ... 2447. 2448. 2449.]]\n\n  [[2450. 2451. 2452. ... 2457. 2458. 2459.]\n   [2460. 2461. 2462. ... 2467. 2468. 2469.]\n   [2470. 2471. 2472. ... 2477. 2478. 2479.]\n   ...\n   [2490. 2491. 2492. ... 2497. 2498. 2499.]\n   [2500. 2501. 2502. ... 2507. 2508. 2509.]\n   [2510. 2511. 2512. ... 2517. 2518. 2519.]]]]\nmultiple_tensor_sum (3, 6, 7, 10)\n[[[[1260. 1262. 1264. ... 1274. 1276. 1278.]\n   [1280. 1282. 1284. ... 1294. 1296. 1298.]\n   [1300. 1302. 1304. ... 1314. 1316. 1318.]\n   ...\n   [1340. 1342. 1344. ... 1354. 1356. 1358.]\n   [1360. 1362. 1364. ... 1374. 1376. 1378.]\n   [1380. 1382. 1384. ... 1394. 1396. 1398.]]\n\n  [[1400. 1402. 1404. ... 1414. 1416. 1418.]\n   [1420. 1422. 1424. ... 1434. 1436. 1438.]\n   [1440. 1442. 1444. ... 1454. 1456. 1458.]\n   ...\n   [1480. 1482. 1484. ... 1494. 1496. 1498.]\n   [1500. 1502. 1504. ... 1514. 1516. 1518.]\n   [1520. 1522. 1524. ... 1534. 1536. 1538.]]\n\n  [[1540. 1542. 1544. ... 1554. 1556. 1558.]\n   [1560. 1562. 1564. ... 1574. 1576. 1578.]\n   [1580. 1582. 1584. ... 1594. 1596. 1598.]\n   ...\n   [1620. 1622. 1624. ... 1634. 1636. 1638.]\n   [1640. 1642. 1644. ... 1654. 1656. 1658.]\n   [1660. 1662. 1664. ... 1674. 1676. 1678.]]\n\n  [[1680. 1682. 1684. ... 1694. 1696. 1698.]\n   [1700. 1702. 1704. ... 1714. 1716. 1718.]\n   [1720. 1722. 1724. ... 1734. 1736. 1738.]\n   ...\n   [1760. 1762. 1764. ... 1774. 1776. 1778.]\n   [1780. 1782. 1784. ... 1794. 1796. 1798.]\n   [1800. 1802. 1804. ... 1814. 1816. 1818.]]\n\n  [[1820. 1822. 1824. ... 1834. 1836. 1838.]\n   [1840. 1842. 1844. ... 1854. 1856. 1858.]\n   [1860. 1862. 1864. ... 1874. 1876. 1878.]\n   ...\n   [1900. 1902. 1904. ... 1914. 1916. 1918.]\n   [1920. 1922. 1924. ... 1934. 1936. 1938.]\n   [1940. 1942. 1944. ... 1954. 1956. 1958.]]\n\n  [[1960. 1962. 1964. ... 1974. 1976. 1978.]\n   [1980. 1982. 1984. ... 1994. 1996. 1998.]\n   [2000. 2002. 2004. ... 2014. 2016. 2018.]\n   ...\n   [2040. 2042. 2044. ... 2054. 2056. 2058.]\n   [2060. 2062. 2064. ... 2074. 2076. 2078.]\n   [2080. 2082. 2084. ... 2094. 2096. 2098.]]]\n\n\n [[[2100. 2102. 2104. ... 2114. 2116. 2118.]\n   [2120. 2122. 2124. ... 2134. 2136. 2138.]\n   [2140. 2142. 2144. ... 2154. 2156. 2158.]\n   ...\n   [2180. 2182. 2184. ... 2194. 2196. 2198.]\n   [2200. 2202. 2204. ... 2214. 2216. 2218.]\n   [2220. 2222. 2224. ... 2234. 2236. 2238.]]\n\n  [[2240. 2242. 2244. ... 2254. 2256. 2258.]\n   [2260. 2262. 2264. ... 2274. 2276. 2278.]\n   [2280. 2282. 2284. ... 2294. 2296. 2298.]\n   ...\n   [2320. 2322. 2324. ... 2334. 2336. 2338.]\n   [2340. 2342. 2344. ... 2354. 2356. 2358.]\n   [2360. 2362. 2364. ... 2374. 2376. 2378.]]\n\n  [[2380. 2382. 2384. ... 2394. 2396. 2398.]\n   [2400. 2402. 2404. ... 2414. 2416. 2418.]\n   [2420. 2422. 2424. ... 2434. 2436. 2438.]\n   ...\n   [2460. 2462. 2464. ... 2474. 2476. 2478.]\n   [2480. 2482. 2484. ... 2494. 2496. 2498.]\n   [2500. 2502. 2504. ... 2514. 2516. 2518.]]\n\n  [[2520. 2522. 2524. ... 2534. 2536. 2538.]\n   [2540. 2542. 2544. ... 2554. 2556. 2558.]\n   [2560. 2562. 2564. ... 2574. 2576. 2578.]\n   ...\n   [2600. 2602. 2604. ... 2614. 2616. 2618.]\n   [2620. 2622. 2624. ... 2634. 2636. 2638.]\n   [2640. 2642. 2644. ... 2654. 2656. 2658.]]\n\n  [[2660. 2662. 2664. ... 2674. 2676. 2678.]\n   [2680. 2682. 2684. ... 2694. 2696. 2698.]\n   [2700. 2702. 2704. ... 2714. 2716. 2718.]\n   ...\n   [2740. 2742. 2744. ... 2754. 2756. 2758.]\n   [2760. 2762. 2764. ... 2774. 2776. 2778.]\n   [2780. 2782. 2784. ... 2794. 2796. 2798.]]\n\n  [[2800. 2802. 2804. ... 2814. 2816. 2818.]\n   [2820. 2822. 2824. ... 2834. 2836. 2838.]\n   [2840. 2842. 2844. ... 2854. 2856. 2858.]\n   ...\n   [2880. 2882. 2884. ... 2894. 2896. 2898.]\n   [2900. 2902. 2904. ... 2914. 2916. 2918.]\n   [2920. 2922. 2924. ... 2934. 2936. 2938.]]]\n\n\n [[[2940. 2942. 2944. ... 2954. 2956. 2958.]\n   [2960. 2962. 2964. ... 2974. 2976. 2978.]\n   [2980. 2982. 2984. ... 2994. 2996. 2998.]\n   ...\n   [3020. 3022. 3024. ... 3034. 3036. 3038.]\n   [3040. 3042. 3044. ... 3054. 3056. 3058.]\n   [3060. 3062. 3064. ... 3074. 3076. 3078.]]\n\n  [[3080. 3082. 3084. ... 3094. 3096. 3098.]\n   [3100. 3102. 3104. ... 3114. 3116. 3118.]\n   [3120. 3122. 3124. ... 3134. 3136. 3138.]\n   ...\n   [3160. 3162. 3164. ... 3174. 3176. 3178.]\n   [3180. 3182. 3184. ... 3194. 3196. 3198.]\n   [3200. 3202. 3204. ... 3214. 3216. 3218.]]\n\n  [[3220. 3222. 3224. ... 3234. 3236. 3238.]\n   [3240. 3242. 3244. ... 3254. 3256. 3258.]\n   [3260. 3262. 3264. ... 3274. 3276. 3278.]\n   ...\n   [3300. 3302. 3304. ... 3314. 3316. 3318.]\n   [3320. 3322. 3324. ... 3334. 3336. 3338.]\n   [3340. 3342. 3344. ... 3354. 3356. 3358.]]\n\n  [[3360. 3362. 3364. ... 3374. 3376. 3378.]\n   [3380. 3382. 3384. ... 3394. 3396. 3398.]\n   [3400. 3402. 3404. ... 3414. 3416. 3418.]\n   ...\n   [3440. 3442. 3444. ... 3454. 3456. 3458.]\n   [3460. 3462. 3464. ... 3474. 3476. 3478.]\n   [3480. 3482. 3484. ... 3494. 3496. 3498.]]\n\n  [[3500. 3502. 3504. ... 3514. 3516. 3518.]\n   [3520. 3522. 3524. ... 3534. 3536. 3538.]\n   [3540. 3542. 3544. ... 3554. 3556. 3558.]\n   ...\n   [3580. 3582. 3584. ... 3594. 3596. 3598.]\n   [3600. 3602. 3604. ... 3614. 3616. 3618.]\n   [3620. 3622. 3624. ... 3634. 3636. 3638.]]\n\n  [[3640. 3642. 3644. ... 3654. 3656. 3658.]\n   [3660. 3662. 3664. ... 3674. 3676. 3678.]\n   [3680. 3682. 3684. ... 3694. 3696. 3698.]\n   ...\n   [3720. 3722. 3724. ... 3734. 3736. 3738.]\n   [3740. 3742. 3744. ... 3754. 3756. 3758.]\n   [3760. 3762. 3764. ... 3774. 3776. 3778.]]]]\n")))}c.isMDXComponent=!0}}]);