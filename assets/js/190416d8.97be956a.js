"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[6139],{3905:function(e,n,r){r.d(n,{Zo:function(){return p},kt:function(){return _}});var t=r(7294);function o(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function a(e,n){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),r.push.apply(r,t)}return r}function s(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{};n%2?a(Object(r),!0).forEach((function(n){o(e,n,r[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(r,n))}))}return e}function i(e,n){if(null==e)return{};var r,t,o=function(e,n){if(null==e)return{};var r,t,o={},a=Object.keys(e);for(t=0;t<a.length;t++)r=a[t],n.indexOf(r)>=0||(o[r]=e[r]);return o}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(t=0;t<a.length;t++)r=a[t],n.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var m=t.createContext({}),c=function(e){var n=t.useContext(m),r=n;return e&&(r="function"==typeof e?e(n):s(s({},n),e)),r},p=function(e){var n=c(e.components);return t.createElement(m.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return t.createElement(t.Fragment,{},n)}},l=t.forwardRef((function(e,n){var r=e.components,o=e.mdxType,a=e.originalType,m=e.parentName,p=i(e,["components","mdxType","originalType","parentName"]),l=c(r),_=o,f=l["".concat(m,".").concat(_)]||l[_]||u[_]||a;return r?t.createElement(f,s(s({ref:n},p),{},{components:r})):t.createElement(f,s({ref:n},p))}));function _(e,n){var r=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var a=r.length,s=new Array(a);s[0]=l;var i={};for(var m in n)hasOwnProperty.call(n,m)&&(i[m]=n[m]);i.originalType=e,i.mdxType="string"==typeof e?e:o,s[1]=i;for(var c=2;c<a;c++)s[c]=r[c];return t.createElement.apply(null,s)}return t.createElement.apply(null,r)}l.displayName="MDXCreateElement"},1643:function(e,n,r){r.r(n),r.d(n,{frontMatter:function(){return i},contentTitle:function(){return m},metadata:function(){return c},toc:function(){return p},default:function(){return l}});var t=r(7462),o=r(3366),a=(r(7294),r(3905)),s=["components"],i={},m=void 0,c={unversionedId:"worked_examples/ex_einsum_new_vector_summation_sparse",id:"worked_examples/ex_einsum_new_vector_summation_sparse",isDocsHomePage:!1,title:"ex_einsum_new_vector_summation_sparse",description:"`py",source:"@site/docs/worked_examples/ex_einsum_new_vector_summation_sparse.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_einsum_new_vector_summation_sparse",permalink:"/csdl/docs/worked_examples/ex_einsum_new_vector_summation_sparse",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_einsum_new_vector_summation_sparse.mdx",tags:[],version:"current",frontMatter:{}},p=[],u={toc:p};function l(e){var n=e.components,r=(0,o.Z)(e,s);return(0,a.kt)("wrapper",(0,t.Z)({},u,r,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nfrom csdl import Model\nimport csdl\n\n\nclass ExampleVectorSummationSparse(Model):\n    def define(self):\n        a = np.arange(4)\n        vec = self.declare_variable('a', val=a)\n\n        self.register_output(\n            'einsum_summ1_sparse_derivs',\n            csdl.einsum_new_api(vec,\n                                operation=[(33, )],\n                                partial_format='sparse'))\n\n\nsim = Simulator(ExampleVectorSummationSparse())\nsim.run()\n\nprint('a', sim['a'].shape)\nprint(sim['a'])\nprint('einsum_summ1_sparse_derivs',\n      sim['einsum_summ1_sparse_derivs'].shape)\nprint(sim['einsum_summ1_sparse_derivs'])\n")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-a",metastring:"(4,)","(4,)":!0},"[0. 1. 2. 3.]\neinsum_summ1_sparse_derivs (1,)\n[6.]\n")))}l.isMDXComponent=!0}}]);