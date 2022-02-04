"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[5806],{3905:function(e,n,r){r.d(n,{Zo:function(){return u},kt:function(){return d}});var t=r(7294);function o(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function a(e,n){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),r.push.apply(r,t)}return r}function c(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{};n%2?a(Object(r),!0).forEach((function(n){o(e,n,r[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(r,n))}))}return e}function i(e,n){if(null==e)return{};var r,t,o=function(e,n){if(null==e)return{};var r,t,o={},a=Object.keys(e);for(t=0;t<a.length;t++)r=a[t],n.indexOf(r)>=0||(o[r]=e[r]);return o}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(t=0;t<a.length;t++)r=a[t],n.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var s=t.createContext({}),l=function(e){var n=t.useContext(s),r=n;return e&&(r="function"==typeof e?e(n):c(c({},n),e)),r},u=function(e){var n=l(e.components);return t.createElement(s.Provider,{value:n},e.children)},m={inlineCode:"code",wrapper:function(e){var n=e.children;return t.createElement(t.Fragment,{},n)}},p=t.forwardRef((function(e,n){var r=e.components,o=e.mdxType,a=e.originalType,s=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),p=l(r),d=o,f=p["".concat(s,".").concat(d)]||p[d]||m[d]||a;return r?t.createElement(f,c(c({ref:n},u),{},{components:r})):t.createElement(f,c({ref:n},u))}));function d(e,n){var r=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var a=r.length,c=new Array(a);c[0]=p;var i={};for(var s in n)hasOwnProperty.call(n,s)&&(i[s]=n[s]);i.originalType=e,i.mdxType="string"==typeof e?e:o,c[1]=i;for(var l=2;l<a;l++)c[l]=r[l];return t.createElement.apply(null,c)}return t.createElement.apply(null,r)}p.displayName="MDXCreateElement"},7896:function(e,n,r){r.r(n),r.d(n,{frontMatter:function(){return i},contentTitle:function(){return s},metadata:function(){return l},toc:function(){return u},default:function(){return p}});var t=r(7462),o=r(3366),a=(r(7294),r(3905)),c=["components"],i={},s=void 0,l={unversionedId:"worked_examples/ex_sum_single_vector_random",id:"worked_examples/ex_sum_single_vector_random",isDocsHomePage:!1,title:"ex_sum_single_vector_random",description:"`py",source:"@site/docs/worked_examples/ex_sum_single_vector_random.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_sum_single_vector_random",permalink:"/csdl/docs/worked_examples/ex_sum_single_vector_random",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_sum_single_vector_random.mdx",tags:[],version:"current",frontMatter:{}},u=[],m={toc:u};function p(e){var n=e.components,r=(0,o.Z)(e,c);return(0,a.kt)("wrapper",(0,t.Z)({},m,r,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleSingleVectorRandom(Model):\n\n    def define(self):\n        n = 3\n        np.random.seed(0)\n\n        # Declare a vector of length 3 as input\n        v1 = self.declare_variable('v1', val=np.random.rand(n))\n\n        # Output the sum of all the elements of the vector v1\n        self.register_output('single_vector_sum', csdl.sum(v1))\n\n\nsim = Simulator(ExampleSingleVectorRandom())\nsim.run()\n\nprint('v1', sim['v1'].shape)\nprint(sim['v1'])\nprint('single_vector_sum', sim['single_vector_sum'].shape)\nprint(sim['single_vector_sum'])\n")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-v1",metastring:"(3,)","(3,)":!0},"[0.5488135  0.71518937 0.60276338]\nsingle_vector_sum (1,)\n[1.86676625]\n")))}p.isMDXComponent=!0}}]);