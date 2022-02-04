"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[5607],{3905:function(n,e,t){t.d(e,{Zo:function(){return c},kt:function(){return _}});var r=t(7294);function o(n,e,t){return e in n?Object.defineProperty(n,e,{value:t,enumerable:!0,configurable:!0,writable:!0}):n[e]=t,n}function a(n,e){var t=Object.keys(n);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(n);e&&(r=r.filter((function(e){return Object.getOwnPropertyDescriptor(n,e).enumerable}))),t.push.apply(t,r)}return t}function s(n){for(var e=1;e<arguments.length;e++){var t=null!=arguments[e]?arguments[e]:{};e%2?a(Object(t),!0).forEach((function(e){o(n,e,t[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(n,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(e){Object.defineProperty(n,e,Object.getOwnPropertyDescriptor(t,e))}))}return n}function i(n,e){if(null==n)return{};var t,r,o=function(n,e){if(null==n)return{};var t,r,o={},a=Object.keys(n);for(r=0;r<a.length;r++)t=a[r],e.indexOf(t)>=0||(o[t]=n[t]);return o}(n,e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(n);for(r=0;r<a.length;r++)t=a[r],e.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(n,t)&&(o[t]=n[t])}return o}var p=r.createContext({}),l=function(n){var e=r.useContext(p),t=e;return n&&(t="function"==typeof n?n(e):s(s({},e),n)),t},c=function(n){var e=l(n.components);return r.createElement(p.Provider,{value:e},n.children)},m={inlineCode:"code",wrapper:function(n){var e=n.children;return r.createElement(r.Fragment,{},e)}},u=r.forwardRef((function(n,e){var t=n.components,o=n.mdxType,a=n.originalType,p=n.parentName,c=i(n,["components","mdxType","originalType","parentName"]),u=l(t),_=o,d=u["".concat(p,".").concat(_)]||u[_]||m[_]||a;return t?r.createElement(d,s(s({ref:e},c),{},{components:t})):r.createElement(d,s({ref:e},c))}));function _(n,e){var t=arguments,o=e&&e.mdxType;if("string"==typeof n||o){var a=t.length,s=new Array(a);s[0]=u;var i={};for(var p in e)hasOwnProperty.call(e,p)&&(i[p]=e[p]);i.originalType=n,i.mdxType="string"==typeof n?n:o,s[1]=i;for(var l=2;l<a;l++)s[l]=t[l];return r.createElement.apply(null,s)}return r.createElement.apply(null,t)}u.displayName="MDXCreateElement"},9836:function(n,e,t){t.r(e),t.d(e,{frontMatter:function(){return i},contentTitle:function(){return p},metadata:function(){return l},toc:function(){return c},default:function(){return u}});var r=t(7462),o=t(3366),a=(t(7294),t(3905)),s=["components"],i={},p=void 0,l={unversionedId:"worked_examples/ex_rotmat_same_radian_tensor_rot_x",id:"worked_examples/ex_rotmat_same_radian_tensor_rot_x",isDocsHomePage:!1,title:"ex_rotmat_same_radian_tensor_rot_x",description:"`py",source:"@site/docs/worked_examples/ex_rotmat_same_radian_tensor_rot_x.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_rotmat_same_radian_tensor_rot_x",permalink:"/csdl/docs/worked_examples/ex_rotmat_same_radian_tensor_rot_x",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_rotmat_same_radian_tensor_rot_x.mdx",tags:[],version:"current",frontMatter:{}},c=[],m={toc:c};function u(n){var e=n.components,t=(0,o.Z)(n,s);return(0,a.kt)("wrapper",(0,r.Z)({},m,t,{components:e,mdxType:"MDXLayout"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleSameRadianTensorRotX(Model):\n\n    def define(self):\n\n        # Shape of a random tensor rotation matrix\n        shape = (2, 3, 4)\n\n        num_elements = np.prod(shape)\n\n        # Tensor of angles in radians\n        angle_val1 = np.repeat(np.pi / 3, num_elements).reshape(shape)\n\n        # Adding the tensor as an input\n        angle_tensor1 = self.declare_variable('tensor', val=angle_val1)\n\n        # Rotation in the x-axis for tensor1\n        self.register_output('tensor_Rot_x',\n                             csdl.rotmat(angle_tensor1, axis='x'))\n\n\nsim = Simulator(ExampleSameRadianTensorRotX())\nsim.run()\n\nprint('tensor', sim['tensor'].shape)\nprint(sim['tensor'])\nprint('tensor_Rot_x', sim['tensor_Rot_x'].shape)\nprint(sim['tensor_Rot_x'])\n")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-tensor",metastring:"(2, 3, 4)","(2,":!0,"3,":!0,"4)":!0},"[[[1.04719755 1.04719755 1.04719755 1.04719755]\n  [1.04719755 1.04719755 1.04719755 1.04719755]\n  [1.04719755 1.04719755 1.04719755 1.04719755]]\n\n [[1.04719755 1.04719755 1.04719755 1.04719755]\n  [1.04719755 1.04719755 1.04719755 1.04719755]\n  [1.04719755 1.04719755 1.04719755 1.04719755]]]\ntensor_Rot_x (2, 3, 4, 3, 3)\n[[[[[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]]\n\n\n  [[[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]]\n\n\n  [[[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]]]\n\n\n\n [[[[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]]\n\n\n  [[[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]]\n\n\n  [[[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]\n\n   [[ 1.         0.         0.       ]\n    [ 0.         0.5       -0.8660254]\n    [ 0.         0.8660254  0.5      ]]]]]\n")))}u.isMDXComponent=!0}}]);