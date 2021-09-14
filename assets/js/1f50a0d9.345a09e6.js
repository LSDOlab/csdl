"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[2793,6139],{3905:function(e,n,r){r.d(n,{Zo:function(){return c},kt:function(){return _}});var t=r(7294);function s(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function a(e,n){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),r.push.apply(r,t)}return r}function i(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{};n%2?a(Object(r),!0).forEach((function(n){s(e,n,r[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(r,n))}))}return e}function o(e,n){if(null==e)return{};var r,t,s=function(e,n){if(null==e)return{};var r,t,s={},a=Object.keys(e);for(t=0;t<a.length;t++)r=a[t],n.indexOf(r)>=0||(s[r]=e[r]);return s}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(t=0;t<a.length;t++)r=a[t],n.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(s[r]=e[r])}return s}var m=t.createContext({}),u=function(e){var n=t.useContext(m),r=n;return e&&(r="function"==typeof e?e(n):i(i({},n),e)),r},c=function(e){var n=u(e.components);return t.createElement(m.Provider,{value:n},e.children)},p={inlineCode:"code",wrapper:function(e){var n=e.children;return t.createElement(t.Fragment,{},n)}},l=t.forwardRef((function(e,n){var r=e.components,s=e.mdxType,a=e.originalType,m=e.parentName,c=o(e,["components","mdxType","originalType","parentName"]),l=u(r),_=s,d=l["".concat(m,".").concat(_)]||l[_]||p[_]||a;return r?t.createElement(d,i(i({ref:n},c),{},{components:r})):t.createElement(d,i({ref:n},c))}));function _(e,n){var r=arguments,s=n&&n.mdxType;if("string"==typeof e||s){var a=r.length,i=new Array(a);i[0]=l;var o={};for(var m in n)hasOwnProperty.call(n,m)&&(o[m]=n[m]);o.originalType=e,o.mdxType="string"==typeof e?e:s,i[1]=o;for(var u=2;u<a;u++)i[u]=r[u];return t.createElement.apply(null,i)}return t.createElement.apply(null,r)}l.displayName="MDXCreateElement"},6914:function(e,n,r){r.r(n),r.d(n,{frontMatter:function(){return m},contentTitle:function(){return u},metadata:function(){return c},toc:function(){return p},default:function(){return _}});var t=r(7462),s=r(3366),a=(r(7294),r(3905)),i=r(1643),o=["components"],m={},u="Single Vector Summation using Einsum with Sparse Partials",c={unversionedId:"examples/Standard Library/einsum_new/ex_einsum_new_vector_summation_sparse",id:"examples/Standard Library/einsum_new/ex_einsum_new_vector_summation_sparse",isDocsHomePage:!1,title:"Single Vector Summation using Einsum with Sparse Partials",description:"This is an example of how to properly use the einsum function",source:"@site/docs/examples/Standard Library/einsum_new/ex_einsum_new_vector_summation_sparse.mdx",sourceDirName:"examples/Standard Library/einsum_new",slug:"/examples/Standard Library/einsum_new/ex_einsum_new_vector_summation_sparse",permalink:"/csdl/docs/examples/Standard Library/einsum_new/ex_einsum_new_vector_summation_sparse",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/einsum_new/ex_einsum_new_vector_summation_sparse.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Single Vector Summation using Einsum",permalink:"/csdl/docs/examples/Standard Library/einsum_new/ex_einsum_new_vector_summation"},next:{title:"Tensor-Vector Inner Product using Einsum",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_inner_tensor_vector"}},p=[],l={toc:p};function _(e){var n=e.components,r=(0,s.Z)(e,o);return(0,a.kt)("wrapper",(0,t.Z)({},l,r,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"single-vector-summation-using-einsum-with-sparse-partials"},"Single Vector Summation using Einsum with Sparse Partials"),(0,a.kt)("p",null,"This is an example of how to properly use the einsum function\nto compute the summation of a single vector while using sparse\npartial derivative."),(0,a.kt)(i.default,{mdxType:"WorkedExample"}))}_.isMDXComponent=!0},1643:function(e,n,r){r.r(n),r.d(n,{frontMatter:function(){return o},contentTitle:function(){return m},metadata:function(){return u},toc:function(){return c},default:function(){return l}});var t=r(7462),s=r(3366),a=(r(7294),r(3905)),i=["components"],o={},m=void 0,u={unversionedId:"worked_examples/ex_einsum_new_vector_summation_sparse",id:"worked_examples/ex_einsum_new_vector_summation_sparse",isDocsHomePage:!1,title:"ex_einsum_new_vector_summation_sparse",description:"`py",source:"@site/docs/worked_examples/ex_einsum_new_vector_summation_sparse.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_einsum_new_vector_summation_sparse",permalink:"/csdl/docs/worked_examples/ex_einsum_new_vector_summation_sparse",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_einsum_new_vector_summation_sparse.mdx",tags:[],version:"current",frontMatter:{}},c=[],p={toc:c};function l(e){var n=e.components,r=(0,s.Z)(e,i);return(0,a.kt)("wrapper",(0,t.Z)({},p,r,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nfrom csdl import Model\nimport csdl\n\n\nclass ExampleVectorSummationSparse(Model):\n    def define(self):\n        a = np.arange(4)\n        vec = self.declare_variable('a', val=a)\n\n        self.register_output(\n            'einsum_summ1_sparse_derivs',\n            csdl.einsum_new_api(vec,\n                                operation=[(33, )],\n                                partial_format='sparse'))\n\n\nsim = Simulator(ExampleVectorSummationSparse())\nsim.run()\n\nprint('a', sim['a'].shape)\nprint(sim['a'])\nprint('einsum_summ1_sparse_derivs',\n      sim['einsum_summ1_sparse_derivs'].shape)\nprint(sim['einsum_summ1_sparse_derivs'])\n")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-a",metastring:"(4,)","(4,)":!0},"[0. 1. 2. 3.]\neinsum_summ1_sparse_derivs (1,)\n[6.]\n")))}l.isMDXComponent=!0}}]);