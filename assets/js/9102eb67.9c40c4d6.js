"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[52],{3905:function(e,t,r){r.d(t,{Zo:function(){return u},kt:function(){return m}});var n=r(7294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var c=n.createContext({}),s=function(e){var t=n.useContext(c),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},u=function(e){var t=s(e.components);return n.createElement(c.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},f=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,o=e.originalType,c=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),f=s(r),m=a,g=f["".concat(c,".").concat(m)]||f[m]||p[m]||o;return r?n.createElement(g,i(i({ref:t},u),{},{components:r})):n.createElement(g,i({ref:t},u))}));function m(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=r.length,i=new Array(o);i[0]=f;var l={};for(var c in t)hasOwnProperty.call(t,c)&&(l[c]=t[c]);l.originalType=e,l.mdxType="string"==typeof e?e:a,i[1]=l;for(var s=2;s<o;s++)i[s]=r[s];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}f.displayName="MDXCreateElement"},2373:function(e,t,r){r.r(t),r.d(t,{frontMatter:function(){return l},contentTitle:function(){return c},metadata:function(){return s},toc:function(){return u},default:function(){return f}});var n=r(7462),a=r(3366),o=(r(7294),r(3905)),i=["components"],l={},c="Average",s={unversionedId:"std_lib_ref/average",id:"std_lib_ref/average",isDocsHomePage:!1,title:"Average",description:"This function allows you to compute the average of vectors, matrices, and tensors.",source:"@site/docs/std_lib_ref/average.mdx",sourceDirName:"std_lib_ref",slug:"/std_lib_ref/average",permalink:"/csdl/docs/std_lib_ref/average",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/std_lib_ref/average.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Rotation Matrix",permalink:"/csdl/docs/std_lib_ref/Vector Algebra/rotmat"},next:{title:"Logarithmic and Exponential Functions",permalink:"/csdl/docs/std_lib_ref/logarithmic_exponentials"}},u=[],p={toc:u};function f(e){var t=e.components,r=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,n.Z)({},p,r,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"average"},"Average"),(0,o.kt)("p",null,"This function allows you to compute the average of vectors, matrices, and tensors."),(0,o.kt)("p",null,"Examples for all the possible use cases are provided below."),(0,o.kt)("p",null,".. autofunction:: csdl.std.average.average"),(0,o.kt)("p",null,"'Single Inputs':","['ex_average_single_vector',\n'ex_average_single_matrix',\n'ex_average_single_tensor',]",","),(0,o.kt)("p",null,"'Multiple Inputs': ","[\n'ex_average_multiple_vector.rst',\n'ex_average_multiple_matrix.rst',\n'ex_average_multiple_tensor.rst',\n]",","),(0,o.kt)("p",null,"'Along an Axis': ","[\n'ex_average_single_matrix_along0',\n'ex_average_single_matrix_along1',\n'ex_average_multiple_matrix_along0',\n'ex_average_multiple_matrix_along1',\n]",","))}f.isMDXComponent=!0}}]);