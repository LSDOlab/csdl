"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[4178,207],{3905:function(t,e,r){r.d(e,{Zo:function(){return m},kt:function(){return d}});var a=r(7294);function n(t,e,r){return e in t?Object.defineProperty(t,e,{value:r,enumerable:!0,configurable:!0,writable:!0}):t[e]=r,t}function o(t,e){var r=Object.keys(t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(t);e&&(a=a.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),r.push.apply(r,a)}return r}function i(t){for(var e=1;e<arguments.length;e++){var r=null!=arguments[e]?arguments[e]:{};e%2?o(Object(r),!0).forEach((function(e){n(t,e,r[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(r,e))}))}return t}function s(t,e){if(null==t)return{};var r,a,n=function(t,e){if(null==t)return{};var r,a,n={},o=Object.keys(t);for(a=0;a<o.length;a++)r=o[a],e.indexOf(r)>=0||(n[r]=t[r]);return n}(t,e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(t);for(a=0;a<o.length;a++)r=o[a],e.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(t,r)&&(n[r]=t[r])}return n}var l=a.createContext({}),c=function(t){var e=a.useContext(l),r=e;return t&&(r="function"==typeof t?t(e):i(i({},e),t)),r},m=function(t){var e=c(t.components);return a.createElement(l.Provider,{value:e},t.children)},p={inlineCode:"code",wrapper:function(t){var e=t.children;return a.createElement(a.Fragment,{},e)}},u=a.forwardRef((function(t,e){var r=t.components,n=t.mdxType,o=t.originalType,l=t.parentName,m=s(t,["components","mdxType","originalType","parentName"]),u=c(r),d=n,_=u["".concat(l,".").concat(d)]||u[d]||p[d]||o;return r?a.createElement(_,i(i({ref:e},m),{},{components:r})):a.createElement(_,i({ref:e},m))}));function d(t,e){var r=arguments,n=e&&e.mdxType;if("string"==typeof t||n){var o=r.length,i=new Array(o);i[0]=u;var s={};for(var l in e)hasOwnProperty.call(e,l)&&(s[l]=e[l]);s.originalType=t,s.mdxType="string"==typeof t?t:n,i[1]=s;for(var c=2;c<o;c++)i[c]=r[c];return a.createElement.apply(null,i)}return a.createElement.apply(null,r)}u.displayName="MDXCreateElement"},6699:function(t,e,r){r.r(e),r.d(e,{frontMatter:function(){return l},contentTitle:function(){return c},metadata:function(){return m},toc:function(){return p},default:function(){return d}});var a=r(7462),n=r(3366),o=(r(7294),r(3905)),i=r(9306),s=["components"],l={},c="Y-Axis Rotation Matrix",m={unversionedId:"examples/Standard Library/rotmat/ex_rotmat_scalar_rot_y",id:"examples/Standard Library/rotmat/ex_rotmat_scalar_rot_y",isDocsHomePage:!1,title:"Y-Axis Rotation Matrix",description:"This example generates a rotation matrix that rotates a vector about the",source:"@site/docs/examples/Standard Library/rotmat/ex_rotmat_scalar_rot_y.mdx",sourceDirName:"examples/Standard Library/rotmat",slug:"/examples/Standard Library/rotmat/ex_rotmat_scalar_rot_y",permalink:"/csdl/docs/examples/Standard Library/rotmat/ex_rotmat_scalar_rot_y",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/rotmat/ex_rotmat_scalar_rot_y.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"X-Axis Rotation Matrix",permalink:"/csdl/docs/examples/Standard Library/rotmat/ex_rotmat_scalar_rot_x"},next:{title:"Sum of Multiple Matrices",permalink:"/csdl/docs/examples/Standard Library/sum/ex_sum_multiple_matrix"}},p=[],u={toc:p};function d(t){var e=t.components,r=(0,n.Z)(t,s);return(0,o.kt)("wrapper",(0,a.Z)({},u,r,{components:e,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"y-axis-rotation-matrix"},"Y-Axis Rotation Matrix"),(0,o.kt)("p",null,"This example generates a rotation matrix that rotates a vector about the\ny-axis by a specified angle."),(0,o.kt)(i.default,{mdxType:"WorkedExample"}))}d.isMDXComponent=!0},9306:function(t,e,r){r.r(e),r.d(e,{frontMatter:function(){return s},contentTitle:function(){return l},metadata:function(){return c},toc:function(){return m},default:function(){return u}});var a=r(7462),n=r(3366),o=(r(7294),r(3905)),i=["components"],s={},l=void 0,c={unversionedId:"worked_examples/ex_rotmat_scalar_rot_y",id:"worked_examples/ex_rotmat_scalar_rot_y",isDocsHomePage:!1,title:"ex_rotmat_scalar_rot_y",description:"`py",source:"@site/docs/worked_examples/ex_rotmat_scalar_rot_y.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_rotmat_scalar_rot_y",permalink:"/csdl/docs/worked_examples/ex_rotmat_scalar_rot_y",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_rotmat_scalar_rot_y.mdx",tags:[],version:"current",frontMatter:{}},m=[],p={toc:m};function u(t){var e=t.components,r=(0,n.Z)(t,i);return(0,o.kt)("wrapper",(0,a.Z)({},p,r,{components:e,mdxType:"MDXLayout"}),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleScalarRotY(Model):\n    def define(self):\n        angle_val3 = np.pi / 3\n\n        angle_scalar = self.declare_variable('scalar', val=angle_val3)\n\n        # Rotation in the y-axis for scalar\n        self.register_output('scalar_Rot_y',\n                             csdl.rotmat(angle_scalar, axis='y'))\n\n\nsim = Simulator(ExampleScalarRotY())\nsim.run()\n\nprint('scalar', sim['scalar'].shape)\nprint(sim['scalar'])\nprint('scalar_Rot_y', sim['scalar_Rot_y'].shape)\nprint(sim['scalar_Rot_y'])\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-scalar",metastring:"(1,)","(1,)":!0},"[1.04719755]\nscalar_Rot_y (3, 3)\n[[ 0.5        0.         0.8660254]\n [ 0.         1.         0.       ]\n [-0.8660254  0.         0.5      ]]\n")))}u.isMDXComponent=!0}}]);