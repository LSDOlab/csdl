"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[1798,9679],{3905:function(e,r,n){n.d(r,{Zo:function(){return l},kt:function(){return m}});var t=n(7294);function o(e,r,n){return r in e?Object.defineProperty(e,r,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[r]=n,e}function a(e,r){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);r&&(t=t.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),n.push.apply(n,t)}return n}function s(e){for(var r=1;r<arguments.length;r++){var n=null!=arguments[r]?arguments[r]:{};r%2?a(Object(n),!0).forEach((function(r){o(e,r,n[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))}))}return e}function i(e,r){if(null==e)return{};var n,t,o=function(e,r){if(null==e)return{};var n,t,o={},a=Object.keys(e);for(t=0;t<a.length;t++)n=a[t],r.indexOf(n)>=0||(o[n]=e[n]);return o}(e,r);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(t=0;t<a.length;t++)n=a[t],r.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var p=t.createContext({}),c=function(e){var r=t.useContext(p),n=r;return e&&(n="function"==typeof e?e(r):s(s({},r),e)),n},l=function(e){var r=c(e.components);return t.createElement(p.Provider,{value:r},e.children)},u={inlineCode:"code",wrapper:function(e){var r=e.children;return t.createElement(t.Fragment,{},r)}},d=t.forwardRef((function(e,r){var n=e.components,o=e.mdxType,a=e.originalType,p=e.parentName,l=i(e,["components","mdxType","originalType","parentName"]),d=c(n),m=o,f=d["".concat(p,".").concat(m)]||d[m]||u[m]||a;return n?t.createElement(f,s(s({ref:r},l),{},{components:n})):t.createElement(f,s({ref:r},l))}));function m(e,r){var n=arguments,o=r&&r.mdxType;if("string"==typeof e||o){var a=n.length,s=new Array(a);s[0]=d;var i={};for(var p in r)hasOwnProperty.call(r,p)&&(i[p]=r[p]);i.originalType=e,i.mdxType="string"==typeof e?e:o,s[1]=i;for(var c=2;c<a;c++)s[c]=n[c];return t.createElement.apply(null,s)}return t.createElement.apply(null,n)}d.displayName="MDXCreateElement"},9515:function(e,r,n){n.r(r),n.d(r,{frontMatter:function(){return p},contentTitle:function(){return c},metadata:function(){return l},toc:function(){return u},default:function(){return m}});var t=n(7462),o=n(3366),a=(n(7294),n(3905)),s=n(1576),i=["components"],p={},c="Reshape a Vector to a Tensor",l={unversionedId:"examples/Standard Library/reshape/ex_reshape_vector2_tensor",id:"examples/Standard Library/reshape/ex_reshape_vector2_tensor",isDocsHomePage:!1,title:"Reshape a Vector to a Tensor",description:"This is an example of how to use the csdl.reshape() function to reshape",source:"@site/docs/examples/Standard Library/reshape/ex_reshape_vector2_tensor.mdx",sourceDirName:"examples/Standard Library/reshape",slug:"/examples/Standard Library/reshape/ex_reshape_vector2_tensor",permalink:"/csdl/docs/examples/Standard Library/reshape/ex_reshape_vector2_tensor",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/reshape/ex_reshape_vector2_tensor.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Reshape a Tensor to a Vector",permalink:"/csdl/docs/examples/Standard Library/reshape/ex_reshape_tensor2_vector"},next:{title:"X-Axis Rotation Tensor For Different Angles in a Tensor",permalink:"/csdl/docs/examples/Standard Library/rotmat/ex_rotmat_diff_radian_tensor_rot_x"}},u=[],d={toc:u};function m(e){var r=e.components,n=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,t.Z)({},d,n,{components:r,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"reshape-a-vector-to-a-tensor"},"Reshape a Vector to a Tensor"),(0,a.kt)("p",null,"This is an example of how to use the csdl.reshape() function to reshape\na vector to a tensor."),(0,a.kt)(s.default,{mdxType:"WorkedExample"}))}m.isMDXComponent=!0},1576:function(e,r,n){n.r(r),n.d(r,{frontMatter:function(){return i},contentTitle:function(){return p},metadata:function(){return c},toc:function(){return l},default:function(){return d}});var t=n(7462),o=n(3366),a=(n(7294),n(3905)),s=["components"],i={},p=void 0,c={unversionedId:"worked_examples/ex_reshape_vector2_tensor",id:"worked_examples/ex_reshape_vector2_tensor",isDocsHomePage:!1,title:"ex_reshape_vector2_tensor",description:"`py",source:"@site/docs/worked_examples/ex_reshape_vector2_tensor.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_reshape_vector2_tensor",permalink:"/csdl/docs/worked_examples/ex_reshape_vector2_tensor",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_reshape_vector2_tensor.mdx",tags:[],version:"current",frontMatter:{}},l=[],u={toc:l};function d(e){var r=e.components,n=(0,o.Z)(e,s);return(0,a.kt)("wrapper",(0,t.Z)({},u,n,{components:r,mdxType:"MDXLayout"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleVector2Tensor(Model):\n    def define(self):\n        shape = (2, 3, 4, 5)\n        size = 2 * 3 * 4 * 5\n\n        # Both vector or tensors need to be numpy arrays\n        tensor = np.arange(size).reshape(shape)\n        vector = np.arange(size)\n\n        # in2 is a vector having a size of 2 * 3 * 4 * 5\n        in2 = self.declare_variable('in2', val=vector)\n\n        # in2 is being reshaped from size =  2 * 3 * 4 * 5 to a ten\xdfsor\n        # having shape = (2, 3, 4, 5)\n        self.register_output('reshape_vector2tensor',\n                             csdl.reshape(in2, new_shape=shape))\n\n\nsim = Simulator(ExampleVector2Tensor())\nsim.run()\n\nprint('in2', sim['in2'].shape)\nprint(sim['in2'])\nprint('reshape_vector2tensor', sim['reshape_vector2tensor'].shape)\nprint(sim['reshape_vector2tensor'])\n")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-in2",metastring:"(120,)","(120,)":!0},"[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.\n  14.  15.  16.  17.  18.  19.  20.  21.  22.  23.  24.  25.  26.  27.\n  28.  29.  30.  31.  32.  33.  34.  35.  36.  37.  38.  39.  40.  41.\n  42.  43.  44.  45.  46.  47.  48.  49.  50.  51.  52.  53.  54.  55.\n  56.  57.  58.  59.  60.  61.  62.  63.  64.  65.  66.  67.  68.  69.\n  70.  71.  72.  73.  74.  75.  76.  77.  78.  79.  80.  81.  82.  83.\n  84.  85.  86.  87.  88.  89.  90.  91.  92.  93.  94.  95.  96.  97.\n  98.  99. 100. 101. 102. 103. 104. 105. 106. 107. 108. 109. 110. 111.\n 112. 113. 114. 115. 116. 117. 118. 119.]\nreshape_vector2tensor (2, 3, 4, 5)\n[[[[  0.   1.   2.   3.   4.]\n   [  5.   6.   7.   8.   9.]\n   [ 10.  11.  12.  13.  14.]\n   [ 15.  16.  17.  18.  19.]]\n\n  [[ 20.  21.  22.  23.  24.]\n   [ 25.  26.  27.  28.  29.]\n   [ 30.  31.  32.  33.  34.]\n   [ 35.  36.  37.  38.  39.]]\n\n  [[ 40.  41.  42.  43.  44.]\n   [ 45.  46.  47.  48.  49.]\n   [ 50.  51.  52.  53.  54.]\n   [ 55.  56.  57.  58.  59.]]]\n\n\n [[[ 60.  61.  62.  63.  64.]\n   [ 65.  66.  67.  68.  69.]\n   [ 70.  71.  72.  73.  74.]\n   [ 75.  76.  77.  78.  79.]]\n\n  [[ 80.  81.  82.  83.  84.]\n   [ 85.  86.  87.  88.  89.]\n   [ 90.  91.  92.  93.  94.]\n   [ 95.  96.  97.  98.  99.]]\n\n  [[100. 101. 102. 103. 104.]\n   [105. 106. 107. 108. 109.]\n   [110. 111. 112. 113. 114.]\n   [115. 116. 117. 118. 119.]]]]\n")))}d.isMDXComponent=!0}}]);