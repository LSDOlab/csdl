"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[611],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return f}});var r=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function l(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?l(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):l(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},l=Object.keys(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var c=r.createContext({}),u=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},p=function(e){var t=u(e.components);return r.createElement(c.Provider,{value:t},e.children)},s={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,l=e.originalType,c=e.parentName,p=i(e,["components","mdxType","originalType","parentName"]),m=u(n),f=o,v=m["".concat(c,".").concat(f)]||m[f]||s[f]||l;return n?r.createElement(v,a(a({ref:t},p),{},{components:n})):r.createElement(v,a({ref:t},p))}));function f(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var l=n.length,a=new Array(l);a[0]=m;var i={};for(var c in t)hasOwnProperty.call(t,c)&&(i[c]=t[c]);i.originalType=e,i.mdxType="string"==typeof e?e:o,a[1]=i;for(var u=2;u<l;u++)a[u]=n[u];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},1364:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return i},contentTitle:function(){return c},metadata:function(){return u},toc:function(){return p},default:function(){return m}});var r=n(7462),o=n(3366),l=(n(7294),n(3905)),a=["components"],i={},c=void 0,u={unversionedId:"worked_examples/ex_sum_multiple_vector",id:"worked_examples/ex_sum_multiple_vector",isDocsHomePage:!1,title:"ex_sum_multiple_vector",description:"`py",source:"@site/docs/worked_examples/ex_sum_multiple_vector.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_sum_multiple_vector",permalink:"/csdl/docs/worked_examples/ex_sum_multiple_vector",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_sum_multiple_vector.mdx",tags:[],version:"current",frontMatter:{}},p=[],s={toc:p};function m(e){var t=e.components,n=(0,o.Z)(e,a);return(0,l.kt)("wrapper",(0,r.Z)({},s,n,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleMultipleVector(Model):\n    def define(self):\n        n = 3\n\n        # Declare a vector of length 3 as input\n        v1 = self.declare_variable('v1', val=np.arange(n))\n\n        # Declare another vector of length 3 as input\n        v2 = self.declare_variable('v2', val=np.arange(n, 2 * n))\n\n        # Output the elementwise sum of vectors v1 and v2\n        self.register_output('multiple_vector_sum', csdl.sum(v1, v2))\n\n\nsim = Simulator(ExampleMultipleVector())\nsim.run()\n\nprint('v1', sim['v1'].shape)\nprint(sim['v1'])\nprint('v2', sim['v2'].shape)\nprint(sim['v2'])\nprint('multiple_vector_sum', sim['multiple_vector_sum'].shape)\nprint(sim['multiple_vector_sum'])\n")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-v1",metastring:"(3,)","(3,)":!0},"[0. 1. 2.]\nv2 (3,)\n[3. 4. 5.]\nmultiple_vector_sum (3,)\n[3. 5. 7.]\n")))}m.isMDXComponent=!0}}]);