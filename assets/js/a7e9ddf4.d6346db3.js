"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[7531,4924],{3905:function(e,n,r){r.d(n,{Zo:function(){return u},kt:function(){return d}});var t=r(7294);function a(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function o(e,n){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),r.push.apply(r,t)}return r}function l(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{};n%2?o(Object(r),!0).forEach((function(n){a(e,n,r[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(r,n))}))}return e}function i(e,n){if(null==e)return{};var r,t,a=function(e,n){if(null==e)return{};var r,t,a={},o=Object.keys(e);for(t=0;t<o.length;t++)r=o[t],n.indexOf(r)>=0||(a[r]=e[r]);return a}(e,n);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(t=0;t<o.length;t++)r=o[t],n.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var s=t.createContext({}),p=function(e){var n=t.useContext(s),r=n;return e&&(r="function"==typeof e?e(n):l(l({},n),e)),r},u=function(e){var n=p(e.components);return t.createElement(s.Provider,{value:n},e.children)},c={inlineCode:"code",wrapper:function(e){var n=e.children;return t.createElement(t.Fragment,{},n)}},m=t.forwardRef((function(e,n){var r=e.components,a=e.mdxType,o=e.originalType,s=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),m=p(r),d=a,f=m["".concat(s,".").concat(d)]||m[d]||c[d]||o;return r?t.createElement(f,l(l({ref:n},u),{},{components:r})):t.createElement(f,l({ref:n},u))}));function d(e,n){var r=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var o=r.length,l=new Array(o);l[0]=m;var i={};for(var s in n)hasOwnProperty.call(n,s)&&(i[s]=n[s]);i.originalType=e,i.mdxType="string"==typeof e?e:a,l[1]=i;for(var p=2;p<o;p++)l[p]=r[p];return t.createElement.apply(null,l)}return t.createElement.apply(null,r)}m.displayName="MDXCreateElement"},6333:function(e,n,r){r.r(n),r.d(n,{frontMatter:function(){return s},contentTitle:function(){return p},metadata:function(){return u},toc:function(){return c},default:function(){return d}});var t=r(7462),a=r(3366),o=(r(7294),r(3905)),l=r(6915),i=["components"],s={},p="Average of Multiple Tensors",u={unversionedId:"examples/Standard Library/average/ex_average_multiple_tensor",id:"examples/Standard Library/average/ex_average_multiple_tensor",isDocsHomePage:!1,title:"Average of Multiple Tensors",description:"This is an example of computing the average of a multiple tensor inputs.",source:"@site/docs/examples/Standard Library/average/ex_average_multiple_tensor.mdx",sourceDirName:"examples/Standard Library/average",slug:"/examples/Standard Library/average/ex_average_multiple_tensor",permalink:"/csdl/docs/examples/Standard Library/average/ex_average_multiple_tensor",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/average/ex_average_multiple_tensor.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Average of Multiple Matrices along Rows",permalink:"/csdl/docs/examples/Standard Library/average/ex_average_multiple_matrix_along1"},next:{title:"Average of Multiple Vectors",permalink:"/csdl/docs/examples/Standard Library/average/ex_average_multiple_vector"}},c=[],m={toc:c};function d(e){var n=e.components,r=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,t.Z)({},m,r,{components:n,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"average-of-multiple-tensors"},"Average of Multiple Tensors"),(0,o.kt)("p",null,"This is an example of computing the average of a multiple tensor inputs."),(0,o.kt)(l.default,{mdxType:"WorkedExample"}))}d.isMDXComponent=!0},6915:function(e,n,r){r.r(n),r.d(n,{frontMatter:function(){return i},contentTitle:function(){return s},metadata:function(){return p},toc:function(){return u},default:function(){return m}});var t=r(7462),a=r(3366),o=(r(7294),r(3905)),l=["components"],i={},s=void 0,p={unversionedId:"worked_examples/ex_average_multiple_tensor",id:"worked_examples/ex_average_multiple_tensor",isDocsHomePage:!1,title:"ex_average_multiple_tensor",description:"`py",source:"@site/docs/worked_examples/ex_average_multiple_tensor.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_average_multiple_tensor",permalink:"/csdl/docs/worked_examples/ex_average_multiple_tensor",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_average_multiple_tensor.mdx",tags:[],version:"current",frontMatter:{}},u=[],c={toc:u};function m(e){var n=e.components,r=(0,a.Z)(e,l);return(0,o.kt)("wrapper",(0,t.Z)({},c,r,{components:n,mdxType:"MDXLayout"}),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleMultipleTensor(Model):\n    def define(self):\n        n = 3\n        m = 6\n        p = 7\n        q = 10\n\n        # Declare a tensor of shape 3x6x7x10 as input\n        T1 = self.declare_variable(\n            'T1',\n            val=np.arange(n * m * p * q).reshape((n, m, p, q)),\n        )\n\n        # Declare another tensor of shape 3x6x7x10 as input\n        T2 = self.declare_variable(\n            'T2',\n            val=np.arange(n * m * p * q, 2 * n * m * p * q).reshape(\n                (n, m, p, q)),\n        )\n        # Output the elementwise average of tensors T1 and T2\n        self.register_output('multiple_tensor_average',\n                             csdl.average(T1, T2))\n\n\nsim = Simulator(ExampleMultipleTensor())\nsim.run()\n\nprint('T1', sim['T1'].shape)\nprint(sim['T1'])\nprint('T2', sim['T2'].shape)\nprint(sim['T2'])\nprint('multiple_tensor_average', sim['multiple_tensor_average'].shape)\nprint(sim['multiple_tensor_average'])\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-T1",metastring:"(3, 6, 7, 10)","(3,":!0,"6,":!0,"7,":!0,"10)":!0},"[[[[0.000e+00 1.000e+00 2.000e+00 ... 7.000e+00 8.000e+00 9.000e+00]\n   [1.000e+01 1.100e+01 1.200e+01 ... 1.700e+01 1.800e+01 1.900e+01]\n   [2.000e+01 2.100e+01 2.200e+01 ... 2.700e+01 2.800e+01 2.900e+01]\n   ...\n   [4.000e+01 4.100e+01 4.200e+01 ... 4.700e+01 4.800e+01 4.900e+01]\n   [5.000e+01 5.100e+01 5.200e+01 ... 5.700e+01 5.800e+01 5.900e+01]\n   [6.000e+01 6.100e+01 6.200e+01 ... 6.700e+01 6.800e+01 6.900e+01]]\n\n  [[7.000e+01 7.100e+01 7.200e+01 ... 7.700e+01 7.800e+01 7.900e+01]\n   [8.000e+01 8.100e+01 8.200e+01 ... 8.700e+01 8.800e+01 8.900e+01]\n   [9.000e+01 9.100e+01 9.200e+01 ... 9.700e+01 9.800e+01 9.900e+01]\n   ...\n   [1.100e+02 1.110e+02 1.120e+02 ... 1.170e+02 1.180e+02 1.190e+02]\n   [1.200e+02 1.210e+02 1.220e+02 ... 1.270e+02 1.280e+02 1.290e+02]\n   [1.300e+02 1.310e+02 1.320e+02 ... 1.370e+02 1.380e+02 1.390e+02]]\n\n  [[1.400e+02 1.410e+02 1.420e+02 ... 1.470e+02 1.480e+02 1.490e+02]\n   [1.500e+02 1.510e+02 1.520e+02 ... 1.570e+02 1.580e+02 1.590e+02]\n   [1.600e+02 1.610e+02 1.620e+02 ... 1.670e+02 1.680e+02 1.690e+02]\n   ...\n   [1.800e+02 1.810e+02 1.820e+02 ... 1.870e+02 1.880e+02 1.890e+02]\n   [1.900e+02 1.910e+02 1.920e+02 ... 1.970e+02 1.980e+02 1.990e+02]\n   [2.000e+02 2.010e+02 2.020e+02 ... 2.070e+02 2.080e+02 2.090e+02]]\n\n  [[2.100e+02 2.110e+02 2.120e+02 ... 2.170e+02 2.180e+02 2.190e+02]\n   [2.200e+02 2.210e+02 2.220e+02 ... 2.270e+02 2.280e+02 2.290e+02]\n   [2.300e+02 2.310e+02 2.320e+02 ... 2.370e+02 2.380e+02 2.390e+02]\n   ...\n   [2.500e+02 2.510e+02 2.520e+02 ... 2.570e+02 2.580e+02 2.590e+02]\n   [2.600e+02 2.610e+02 2.620e+02 ... 2.670e+02 2.680e+02 2.690e+02]\n   [2.700e+02 2.710e+02 2.720e+02 ... 2.770e+02 2.780e+02 2.790e+02]]\n\n  [[2.800e+02 2.810e+02 2.820e+02 ... 2.870e+02 2.880e+02 2.890e+02]\n   [2.900e+02 2.910e+02 2.920e+02 ... 2.970e+02 2.980e+02 2.990e+02]\n   [3.000e+02 3.010e+02 3.020e+02 ... 3.070e+02 3.080e+02 3.090e+02]\n   ...\n   [3.200e+02 3.210e+02 3.220e+02 ... 3.270e+02 3.280e+02 3.290e+02]\n   [3.300e+02 3.310e+02 3.320e+02 ... 3.370e+02 3.380e+02 3.390e+02]\n   [3.400e+02 3.410e+02 3.420e+02 ... 3.470e+02 3.480e+02 3.490e+02]]\n\n  [[3.500e+02 3.510e+02 3.520e+02 ... 3.570e+02 3.580e+02 3.590e+02]\n   [3.600e+02 3.610e+02 3.620e+02 ... 3.670e+02 3.680e+02 3.690e+02]\n   [3.700e+02 3.710e+02 3.720e+02 ... 3.770e+02 3.780e+02 3.790e+02]\n   ...\n   [3.900e+02 3.910e+02 3.920e+02 ... 3.970e+02 3.980e+02 3.990e+02]\n   [4.000e+02 4.010e+02 4.020e+02 ... 4.070e+02 4.080e+02 4.090e+02]\n   [4.100e+02 4.110e+02 4.120e+02 ... 4.170e+02 4.180e+02 4.190e+02]]]\n\n\n [[[4.200e+02 4.210e+02 4.220e+02 ... 4.270e+02 4.280e+02 4.290e+02]\n   [4.300e+02 4.310e+02 4.320e+02 ... 4.370e+02 4.380e+02 4.390e+02]\n   [4.400e+02 4.410e+02 4.420e+02 ... 4.470e+02 4.480e+02 4.490e+02]\n   ...\n   [4.600e+02 4.610e+02 4.620e+02 ... 4.670e+02 4.680e+02 4.690e+02]\n   [4.700e+02 4.710e+02 4.720e+02 ... 4.770e+02 4.780e+02 4.790e+02]\n   [4.800e+02 4.810e+02 4.820e+02 ... 4.870e+02 4.880e+02 4.890e+02]]\n\n  [[4.900e+02 4.910e+02 4.920e+02 ... 4.970e+02 4.980e+02 4.990e+02]\n   [5.000e+02 5.010e+02 5.020e+02 ... 5.070e+02 5.080e+02 5.090e+02]\n   [5.100e+02 5.110e+02 5.120e+02 ... 5.170e+02 5.180e+02 5.190e+02]\n   ...\n   [5.300e+02 5.310e+02 5.320e+02 ... 5.370e+02 5.380e+02 5.390e+02]\n   [5.400e+02 5.410e+02 5.420e+02 ... 5.470e+02 5.480e+02 5.490e+02]\n   [5.500e+02 5.510e+02 5.520e+02 ... 5.570e+02 5.580e+02 5.590e+02]]\n\n  [[5.600e+02 5.610e+02 5.620e+02 ... 5.670e+02 5.680e+02 5.690e+02]\n   [5.700e+02 5.710e+02 5.720e+02 ... 5.770e+02 5.780e+02 5.790e+02]\n   [5.800e+02 5.810e+02 5.820e+02 ... 5.870e+02 5.880e+02 5.890e+02]\n   ...\n   [6.000e+02 6.010e+02 6.020e+02 ... 6.070e+02 6.080e+02 6.090e+02]\n   [6.100e+02 6.110e+02 6.120e+02 ... 6.170e+02 6.180e+02 6.190e+02]\n   [6.200e+02 6.210e+02 6.220e+02 ... 6.270e+02 6.280e+02 6.290e+02]]\n\n  [[6.300e+02 6.310e+02 6.320e+02 ... 6.370e+02 6.380e+02 6.390e+02]\n   [6.400e+02 6.410e+02 6.420e+02 ... 6.470e+02 6.480e+02 6.490e+02]\n   [6.500e+02 6.510e+02 6.520e+02 ... 6.570e+02 6.580e+02 6.590e+02]\n   ...\n   [6.700e+02 6.710e+02 6.720e+02 ... 6.770e+02 6.780e+02 6.790e+02]\n   [6.800e+02 6.810e+02 6.820e+02 ... 6.870e+02 6.880e+02 6.890e+02]\n   [6.900e+02 6.910e+02 6.920e+02 ... 6.970e+02 6.980e+02 6.990e+02]]\n\n  [[7.000e+02 7.010e+02 7.020e+02 ... 7.070e+02 7.080e+02 7.090e+02]\n   [7.100e+02 7.110e+02 7.120e+02 ... 7.170e+02 7.180e+02 7.190e+02]\n   [7.200e+02 7.210e+02 7.220e+02 ... 7.270e+02 7.280e+02 7.290e+02]\n   ...\n   [7.400e+02 7.410e+02 7.420e+02 ... 7.470e+02 7.480e+02 7.490e+02]\n   [7.500e+02 7.510e+02 7.520e+02 ... 7.570e+02 7.580e+02 7.590e+02]\n   [7.600e+02 7.610e+02 7.620e+02 ... 7.670e+02 7.680e+02 7.690e+02]]\n\n  [[7.700e+02 7.710e+02 7.720e+02 ... 7.770e+02 7.780e+02 7.790e+02]\n   [7.800e+02 7.810e+02 7.820e+02 ... 7.870e+02 7.880e+02 7.890e+02]\n   [7.900e+02 7.910e+02 7.920e+02 ... 7.970e+02 7.980e+02 7.990e+02]\n   ...\n   [8.100e+02 8.110e+02 8.120e+02 ... 8.170e+02 8.180e+02 8.190e+02]\n   [8.200e+02 8.210e+02 8.220e+02 ... 8.270e+02 8.280e+02 8.290e+02]\n   [8.300e+02 8.310e+02 8.320e+02 ... 8.370e+02 8.380e+02 8.390e+02]]]\n\n\n [[[8.400e+02 8.410e+02 8.420e+02 ... 8.470e+02 8.480e+02 8.490e+02]\n   [8.500e+02 8.510e+02 8.520e+02 ... 8.570e+02 8.580e+02 8.590e+02]\n   [8.600e+02 8.610e+02 8.620e+02 ... 8.670e+02 8.680e+02 8.690e+02]\n   ...\n   [8.800e+02 8.810e+02 8.820e+02 ... 8.870e+02 8.880e+02 8.890e+02]\n   [8.900e+02 8.910e+02 8.920e+02 ... 8.970e+02 8.980e+02 8.990e+02]\n   [9.000e+02 9.010e+02 9.020e+02 ... 9.070e+02 9.080e+02 9.090e+02]]\n\n  [[9.100e+02 9.110e+02 9.120e+02 ... 9.170e+02 9.180e+02 9.190e+02]\n   [9.200e+02 9.210e+02 9.220e+02 ... 9.270e+02 9.280e+02 9.290e+02]\n   [9.300e+02 9.310e+02 9.320e+02 ... 9.370e+02 9.380e+02 9.390e+02]\n   ...\n   [9.500e+02 9.510e+02 9.520e+02 ... 9.570e+02 9.580e+02 9.590e+02]\n   [9.600e+02 9.610e+02 9.620e+02 ... 9.670e+02 9.680e+02 9.690e+02]\n   [9.700e+02 9.710e+02 9.720e+02 ... 9.770e+02 9.780e+02 9.790e+02]]\n\n  [[9.800e+02 9.810e+02 9.820e+02 ... 9.870e+02 9.880e+02 9.890e+02]\n   [9.900e+02 9.910e+02 9.920e+02 ... 9.970e+02 9.980e+02 9.990e+02]\n   [1.000e+03 1.001e+03 1.002e+03 ... 1.007e+03 1.008e+03 1.009e+03]\n   ...\n   [1.020e+03 1.021e+03 1.022e+03 ... 1.027e+03 1.028e+03 1.029e+03]\n   [1.030e+03 1.031e+03 1.032e+03 ... 1.037e+03 1.038e+03 1.039e+03]\n   [1.040e+03 1.041e+03 1.042e+03 ... 1.047e+03 1.048e+03 1.049e+03]]\n\n  [[1.050e+03 1.051e+03 1.052e+03 ... 1.057e+03 1.058e+03 1.059e+03]\n   [1.060e+03 1.061e+03 1.062e+03 ... 1.067e+03 1.068e+03 1.069e+03]\n   [1.070e+03 1.071e+03 1.072e+03 ... 1.077e+03 1.078e+03 1.079e+03]\n   ...\n   [1.090e+03 1.091e+03 1.092e+03 ... 1.097e+03 1.098e+03 1.099e+03]\n   [1.100e+03 1.101e+03 1.102e+03 ... 1.107e+03 1.108e+03 1.109e+03]\n   [1.110e+03 1.111e+03 1.112e+03 ... 1.117e+03 1.118e+03 1.119e+03]]\n\n  [[1.120e+03 1.121e+03 1.122e+03 ... 1.127e+03 1.128e+03 1.129e+03]\n   [1.130e+03 1.131e+03 1.132e+03 ... 1.137e+03 1.138e+03 1.139e+03]\n   [1.140e+03 1.141e+03 1.142e+03 ... 1.147e+03 1.148e+03 1.149e+03]\n   ...\n   [1.160e+03 1.161e+03 1.162e+03 ... 1.167e+03 1.168e+03 1.169e+03]\n   [1.170e+03 1.171e+03 1.172e+03 ... 1.177e+03 1.178e+03 1.179e+03]\n   [1.180e+03 1.181e+03 1.182e+03 ... 1.187e+03 1.188e+03 1.189e+03]]\n\n  [[1.190e+03 1.191e+03 1.192e+03 ... 1.197e+03 1.198e+03 1.199e+03]\n   [1.200e+03 1.201e+03 1.202e+03 ... 1.207e+03 1.208e+03 1.209e+03]\n   [1.210e+03 1.211e+03 1.212e+03 ... 1.217e+03 1.218e+03 1.219e+03]\n   ...\n   [1.230e+03 1.231e+03 1.232e+03 ... 1.237e+03 1.238e+03 1.239e+03]\n   [1.240e+03 1.241e+03 1.242e+03 ... 1.247e+03 1.248e+03 1.249e+03]\n   [1.250e+03 1.251e+03 1.252e+03 ... 1.257e+03 1.258e+03 1.259e+03]]]]\nT2 (3, 6, 7, 10)\n[[[[1260. 1261. 1262. ... 1267. 1268. 1269.]\n   [1270. 1271. 1272. ... 1277. 1278. 1279.]\n   [1280. 1281. 1282. ... 1287. 1288. 1289.]\n   ...\n   [1300. 1301. 1302. ... 1307. 1308. 1309.]\n   [1310. 1311. 1312. ... 1317. 1318. 1319.]\n   [1320. 1321. 1322. ... 1327. 1328. 1329.]]\n\n  [[1330. 1331. 1332. ... 1337. 1338. 1339.]\n   [1340. 1341. 1342. ... 1347. 1348. 1349.]\n   [1350. 1351. 1352. ... 1357. 1358. 1359.]\n   ...\n   [1370. 1371. 1372. ... 1377. 1378. 1379.]\n   [1380. 1381. 1382. ... 1387. 1388. 1389.]\n   [1390. 1391. 1392. ... 1397. 1398. 1399.]]\n\n  [[1400. 1401. 1402. ... 1407. 1408. 1409.]\n   [1410. 1411. 1412. ... 1417. 1418. 1419.]\n   [1420. 1421. 1422. ... 1427. 1428. 1429.]\n   ...\n   [1440. 1441. 1442. ... 1447. 1448. 1449.]\n   [1450. 1451. 1452. ... 1457. 1458. 1459.]\n   [1460. 1461. 1462. ... 1467. 1468. 1469.]]\n\n  [[1470. 1471. 1472. ... 1477. 1478. 1479.]\n   [1480. 1481. 1482. ... 1487. 1488. 1489.]\n   [1490. 1491. 1492. ... 1497. 1498. 1499.]\n   ...\n   [1510. 1511. 1512. ... 1517. 1518. 1519.]\n   [1520. 1521. 1522. ... 1527. 1528. 1529.]\n   [1530. 1531. 1532. ... 1537. 1538. 1539.]]\n\n  [[1540. 1541. 1542. ... 1547. 1548. 1549.]\n   [1550. 1551. 1552. ... 1557. 1558. 1559.]\n   [1560. 1561. 1562. ... 1567. 1568. 1569.]\n   ...\n   [1580. 1581. 1582. ... 1587. 1588. 1589.]\n   [1590. 1591. 1592. ... 1597. 1598. 1599.]\n   [1600. 1601. 1602. ... 1607. 1608. 1609.]]\n\n  [[1610. 1611. 1612. ... 1617. 1618. 1619.]\n   [1620. 1621. 1622. ... 1627. 1628. 1629.]\n   [1630. 1631. 1632. ... 1637. 1638. 1639.]\n   ...\n   [1650. 1651. 1652. ... 1657. 1658. 1659.]\n   [1660. 1661. 1662. ... 1667. 1668. 1669.]\n   [1670. 1671. 1672. ... 1677. 1678. 1679.]]]\n\n\n [[[1680. 1681. 1682. ... 1687. 1688. 1689.]\n   [1690. 1691. 1692. ... 1697. 1698. 1699.]\n   [1700. 1701. 1702. ... 1707. 1708. 1709.]\n   ...\n   [1720. 1721. 1722. ... 1727. 1728. 1729.]\n   [1730. 1731. 1732. ... 1737. 1738. 1739.]\n   [1740. 1741. 1742. ... 1747. 1748. 1749.]]\n\n  [[1750. 1751. 1752. ... 1757. 1758. 1759.]\n   [1760. 1761. 1762. ... 1767. 1768. 1769.]\n   [1770. 1771. 1772. ... 1777. 1778. 1779.]\n   ...\n   [1790. 1791. 1792. ... 1797. 1798. 1799.]\n   [1800. 1801. 1802. ... 1807. 1808. 1809.]\n   [1810. 1811. 1812. ... 1817. 1818. 1819.]]\n\n  [[1820. 1821. 1822. ... 1827. 1828. 1829.]\n   [1830. 1831. 1832. ... 1837. 1838. 1839.]\n   [1840. 1841. 1842. ... 1847. 1848. 1849.]\n   ...\n   [1860. 1861. 1862. ... 1867. 1868. 1869.]\n   [1870. 1871. 1872. ... 1877. 1878. 1879.]\n   [1880. 1881. 1882. ... 1887. 1888. 1889.]]\n\n  [[1890. 1891. 1892. ... 1897. 1898. 1899.]\n   [1900. 1901. 1902. ... 1907. 1908. 1909.]\n   [1910. 1911. 1912. ... 1917. 1918. 1919.]\n   ...\n   [1930. 1931. 1932. ... 1937. 1938. 1939.]\n   [1940. 1941. 1942. ... 1947. 1948. 1949.]\n   [1950. 1951. 1952. ... 1957. 1958. 1959.]]\n\n  [[1960. 1961. 1962. ... 1967. 1968. 1969.]\n   [1970. 1971. 1972. ... 1977. 1978. 1979.]\n   [1980. 1981. 1982. ... 1987. 1988. 1989.]\n   ...\n   [2000. 2001. 2002. ... 2007. 2008. 2009.]\n   [2010. 2011. 2012. ... 2017. 2018. 2019.]\n   [2020. 2021. 2022. ... 2027. 2028. 2029.]]\n\n  [[2030. 2031. 2032. ... 2037. 2038. 2039.]\n   [2040. 2041. 2042. ... 2047. 2048. 2049.]\n   [2050. 2051. 2052. ... 2057. 2058. 2059.]\n   ...\n   [2070. 2071. 2072. ... 2077. 2078. 2079.]\n   [2080. 2081. 2082. ... 2087. 2088. 2089.]\n   [2090. 2091. 2092. ... 2097. 2098. 2099.]]]\n\n\n [[[2100. 2101. 2102. ... 2107. 2108. 2109.]\n   [2110. 2111. 2112. ... 2117. 2118. 2119.]\n   [2120. 2121. 2122. ... 2127. 2128. 2129.]\n   ...\n   [2140. 2141. 2142. ... 2147. 2148. 2149.]\n   [2150. 2151. 2152. ... 2157. 2158. 2159.]\n   [2160. 2161. 2162. ... 2167. 2168. 2169.]]\n\n  [[2170. 2171. 2172. ... 2177. 2178. 2179.]\n   [2180. 2181. 2182. ... 2187. 2188. 2189.]\n   [2190. 2191. 2192. ... 2197. 2198. 2199.]\n   ...\n   [2210. 2211. 2212. ... 2217. 2218. 2219.]\n   [2220. 2221. 2222. ... 2227. 2228. 2229.]\n   [2230. 2231. 2232. ... 2237. 2238. 2239.]]\n\n  [[2240. 2241. 2242. ... 2247. 2248. 2249.]\n   [2250. 2251. 2252. ... 2257. 2258. 2259.]\n   [2260. 2261. 2262. ... 2267. 2268. 2269.]\n   ...\n   [2280. 2281. 2282. ... 2287. 2288. 2289.]\n   [2290. 2291. 2292. ... 2297. 2298. 2299.]\n   [2300. 2301. 2302. ... 2307. 2308. 2309.]]\n\n  [[2310. 2311. 2312. ... 2317. 2318. 2319.]\n   [2320. 2321. 2322. ... 2327. 2328. 2329.]\n   [2330. 2331. 2332. ... 2337. 2338. 2339.]\n   ...\n   [2350. 2351. 2352. ... 2357. 2358. 2359.]\n   [2360. 2361. 2362. ... 2367. 2368. 2369.]\n   [2370. 2371. 2372. ... 2377. 2378. 2379.]]\n\n  [[2380. 2381. 2382. ... 2387. 2388. 2389.]\n   [2390. 2391. 2392. ... 2397. 2398. 2399.]\n   [2400. 2401. 2402. ... 2407. 2408. 2409.]\n   ...\n   [2420. 2421. 2422. ... 2427. 2428. 2429.]\n   [2430. 2431. 2432. ... 2437. 2438. 2439.]\n   [2440. 2441. 2442. ... 2447. 2448. 2449.]]\n\n  [[2450. 2451. 2452. ... 2457. 2458. 2459.]\n   [2460. 2461. 2462. ... 2467. 2468. 2469.]\n   [2470. 2471. 2472. ... 2477. 2478. 2479.]\n   ...\n   [2490. 2491. 2492. ... 2497. 2498. 2499.]\n   [2500. 2501. 2502. ... 2507. 2508. 2509.]\n   [2510. 2511. 2512. ... 2517. 2518. 2519.]]]]\nmultiple_tensor_average (3, 6, 7, 10)\n[[[[ 630.  631.  632. ...  637.  638.  639.]\n   [ 640.  641.  642. ...  647.  648.  649.]\n   [ 650.  651.  652. ...  657.  658.  659.]\n   ...\n   [ 670.  671.  672. ...  677.  678.  679.]\n   [ 680.  681.  682. ...  687.  688.  689.]\n   [ 690.  691.  692. ...  697.  698.  699.]]\n\n  [[ 700.  701.  702. ...  707.  708.  709.]\n   [ 710.  711.  712. ...  717.  718.  719.]\n   [ 720.  721.  722. ...  727.  728.  729.]\n   ...\n   [ 740.  741.  742. ...  747.  748.  749.]\n   [ 750.  751.  752. ...  757.  758.  759.]\n   [ 760.  761.  762. ...  767.  768.  769.]]\n\n  [[ 770.  771.  772. ...  777.  778.  779.]\n   [ 780.  781.  782. ...  787.  788.  789.]\n   [ 790.  791.  792. ...  797.  798.  799.]\n   ...\n   [ 810.  811.  812. ...  817.  818.  819.]\n   [ 820.  821.  822. ...  827.  828.  829.]\n   [ 830.  831.  832. ...  837.  838.  839.]]\n\n  [[ 840.  841.  842. ...  847.  848.  849.]\n   [ 850.  851.  852. ...  857.  858.  859.]\n   [ 860.  861.  862. ...  867.  868.  869.]\n   ...\n   [ 880.  881.  882. ...  887.  888.  889.]\n   [ 890.  891.  892. ...  897.  898.  899.]\n   [ 900.  901.  902. ...  907.  908.  909.]]\n\n  [[ 910.  911.  912. ...  917.  918.  919.]\n   [ 920.  921.  922. ...  927.  928.  929.]\n   [ 930.  931.  932. ...  937.  938.  939.]\n   ...\n   [ 950.  951.  952. ...  957.  958.  959.]\n   [ 960.  961.  962. ...  967.  968.  969.]\n   [ 970.  971.  972. ...  977.  978.  979.]]\n\n  [[ 980.  981.  982. ...  987.  988.  989.]\n   [ 990.  991.  992. ...  997.  998.  999.]\n   [1000. 1001. 1002. ... 1007. 1008. 1009.]\n   ...\n   [1020. 1021. 1022. ... 1027. 1028. 1029.]\n   [1030. 1031. 1032. ... 1037. 1038. 1039.]\n   [1040. 1041. 1042. ... 1047. 1048. 1049.]]]\n\n\n [[[1050. 1051. 1052. ... 1057. 1058. 1059.]\n   [1060. 1061. 1062. ... 1067. 1068. 1069.]\n   [1070. 1071. 1072. ... 1077. 1078. 1079.]\n   ...\n   [1090. 1091. 1092. ... 1097. 1098. 1099.]\n   [1100. 1101. 1102. ... 1107. 1108. 1109.]\n   [1110. 1111. 1112. ... 1117. 1118. 1119.]]\n\n  [[1120. 1121. 1122. ... 1127. 1128. 1129.]\n   [1130. 1131. 1132. ... 1137. 1138. 1139.]\n   [1140. 1141. 1142. ... 1147. 1148. 1149.]\n   ...\n   [1160. 1161. 1162. ... 1167. 1168. 1169.]\n   [1170. 1171. 1172. ... 1177. 1178. 1179.]\n   [1180. 1181. 1182. ... 1187. 1188. 1189.]]\n\n  [[1190. 1191. 1192. ... 1197. 1198. 1199.]\n   [1200. 1201. 1202. ... 1207. 1208. 1209.]\n   [1210. 1211. 1212. ... 1217. 1218. 1219.]\n   ...\n   [1230. 1231. 1232. ... 1237. 1238. 1239.]\n   [1240. 1241. 1242. ... 1247. 1248. 1249.]\n   [1250. 1251. 1252. ... 1257. 1258. 1259.]]\n\n  [[1260. 1261. 1262. ... 1267. 1268. 1269.]\n   [1270. 1271. 1272. ... 1277. 1278. 1279.]\n   [1280. 1281. 1282. ... 1287. 1288. 1289.]\n   ...\n   [1300. 1301. 1302. ... 1307. 1308. 1309.]\n   [1310. 1311. 1312. ... 1317. 1318. 1319.]\n   [1320. 1321. 1322. ... 1327. 1328. 1329.]]\n\n  [[1330. 1331. 1332. ... 1337. 1338. 1339.]\n   [1340. 1341. 1342. ... 1347. 1348. 1349.]\n   [1350. 1351. 1352. ... 1357. 1358. 1359.]\n   ...\n   [1370. 1371. 1372. ... 1377. 1378. 1379.]\n   [1380. 1381. 1382. ... 1387. 1388. 1389.]\n   [1390. 1391. 1392. ... 1397. 1398. 1399.]]\n\n  [[1400. 1401. 1402. ... 1407. 1408. 1409.]\n   [1410. 1411. 1412. ... 1417. 1418. 1419.]\n   [1420. 1421. 1422. ... 1427. 1428. 1429.]\n   ...\n   [1440. 1441. 1442. ... 1447. 1448. 1449.]\n   [1450. 1451. 1452. ... 1457. 1458. 1459.]\n   [1460. 1461. 1462. ... 1467. 1468. 1469.]]]\n\n\n [[[1470. 1471. 1472. ... 1477. 1478. 1479.]\n   [1480. 1481. 1482. ... 1487. 1488. 1489.]\n   [1490. 1491. 1492. ... 1497. 1498. 1499.]\n   ...\n   [1510. 1511. 1512. ... 1517. 1518. 1519.]\n   [1520. 1521. 1522. ... 1527. 1528. 1529.]\n   [1530. 1531. 1532. ... 1537. 1538. 1539.]]\n\n  [[1540. 1541. 1542. ... 1547. 1548. 1549.]\n   [1550. 1551. 1552. ... 1557. 1558. 1559.]\n   [1560. 1561. 1562. ... 1567. 1568. 1569.]\n   ...\n   [1580. 1581. 1582. ... 1587. 1588. 1589.]\n   [1590. 1591. 1592. ... 1597. 1598. 1599.]\n   [1600. 1601. 1602. ... 1607. 1608. 1609.]]\n\n  [[1610. 1611. 1612. ... 1617. 1618. 1619.]\n   [1620. 1621. 1622. ... 1627. 1628. 1629.]\n   [1630. 1631. 1632. ... 1637. 1638. 1639.]\n   ...\n   [1650. 1651. 1652. ... 1657. 1658. 1659.]\n   [1660. 1661. 1662. ... 1667. 1668. 1669.]\n   [1670. 1671. 1672. ... 1677. 1678. 1679.]]\n\n  [[1680. 1681. 1682. ... 1687. 1688. 1689.]\n   [1690. 1691. 1692. ... 1697. 1698. 1699.]\n   [1700. 1701. 1702. ... 1707. 1708. 1709.]\n   ...\n   [1720. 1721. 1722. ... 1727. 1728. 1729.]\n   [1730. 1731. 1732. ... 1737. 1738. 1739.]\n   [1740. 1741. 1742. ... 1747. 1748. 1749.]]\n\n  [[1750. 1751. 1752. ... 1757. 1758. 1759.]\n   [1760. 1761. 1762. ... 1767. 1768. 1769.]\n   [1770. 1771. 1772. ... 1777. 1778. 1779.]\n   ...\n   [1790. 1791. 1792. ... 1797. 1798. 1799.]\n   [1800. 1801. 1802. ... 1807. 1808. 1809.]\n   [1810. 1811. 1812. ... 1817. 1818. 1819.]]\n\n  [[1820. 1821. 1822. ... 1827. 1828. 1829.]\n   [1830. 1831. 1832. ... 1837. 1838. 1839.]\n   [1840. 1841. 1842. ... 1847. 1848. 1849.]\n   ...\n   [1860. 1861. 1862. ... 1867. 1868. 1869.]\n   [1870. 1871. 1872. ... 1877. 1878. 1879.]\n   [1880. 1881. 1882. ... 1887. 1888. 1889.]]]]\n")))}m.isMDXComponent=!0}}]);