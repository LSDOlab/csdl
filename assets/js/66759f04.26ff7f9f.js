"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[606,689],{3905:function(e,n,t){t.d(n,{Zo:function(){return l},kt:function(){return d}});var r=t(7294);function o(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function s(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function i(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?s(Object(t),!0).forEach((function(n){o(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):s(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function a(e,n){if(null==e)return{};var t,r,o=function(e,n){if(null==e)return{};var t,r,o={},s=Object.keys(e);for(r=0;r<s.length;r++)t=s[r],n.indexOf(t)>=0||(o[t]=e[t]);return o}(e,n);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(r=0;r<s.length;r++)t=s[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}var m=r.createContext({}),u=function(e){var n=r.useContext(m),t=n;return e&&(t="function"==typeof e?e(n):i(i({},n),e)),t},l=function(e){var n=u(e.components);return r.createElement(m.Provider,{value:n},e.children)},c={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},p=r.forwardRef((function(e,n){var t=e.components,o=e.mdxType,s=e.originalType,m=e.parentName,l=a(e,["components","mdxType","originalType","parentName"]),p=u(t),d=o,_=p["".concat(m,".").concat(d)]||p[d]||c[d]||s;return t?r.createElement(_,i(i({ref:n},l),{},{components:t})):r.createElement(_,i({ref:n},l))}));function d(e,n){var t=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var s=t.length,i=new Array(s);i[0]=p;var a={};for(var m in n)hasOwnProperty.call(n,m)&&(a[m]=n[m]);a.originalType=e,a.mdxType="string"==typeof e?e:o,i[1]=a;for(var u=2;u<s;u++)i[u]=t[u];return r.createElement.apply(null,i)}return r.createElement.apply(null,t)}p.displayName="MDXCreateElement"},778:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return m},contentTitle:function(){return u},metadata:function(){return l},toc:function(){return c},default:function(){return d}});var r=t(7462),o=t(3366),s=(t(7294),t(3905)),i=t(8742),a=["components"],m={},u="Single Tensor Summation using Einsum",l={unversionedId:"examples/Standard Library/einsum_old/ex_einsum_old_tensor_summation",id:"examples/Standard Library/einsum_old/ex_einsum_old_tensor_summation",isDocsHomePage:!1,title:"Single Tensor Summation using Einsum",description:"This is an example of how to properly use the einsum function",source:"@site/docs/examples/Standard Library/einsum_old/ex_einsum_old_tensor_summation.mdx",sourceDirName:"examples/Standard Library/einsum_old",slug:"/examples/Standard Library/einsum_old/ex_einsum_old_tensor_summation",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_tensor_summation",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/einsum_old/ex_einsum_old_tensor_summation.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Reordering a Tensor using Einsum with Sparse Partials",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_reorder_tensor_sparse"},next:{title:"Single Tensor Summation using Einsum with Sparse Partials",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_tensor_summation_sparse"}},c=[],p={toc:c};function d(e){var n=e.components,t=(0,o.Z)(e,a);return(0,s.kt)("wrapper",(0,r.Z)({},p,t,{components:n,mdxType:"MDXLayout"}),(0,s.kt)("h1",{id:"single-tensor-summation-using-einsum"},"Single Tensor Summation using Einsum"),(0,s.kt)("p",null,"This is an example of how to properly use the einsum function\nto compute the summation of a single tensor."),(0,s.kt)(i.default,{mdxType:"WorkedExample"}))}d.isMDXComponent=!0},8742:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return a},contentTitle:function(){return m},metadata:function(){return u},toc:function(){return l},default:function(){return p}});var r=t(7462),o=t(3366),s=(t(7294),t(3905)),i=["components"],a={},m=void 0,u={unversionedId:"worked_examples/ex_einsum_old_tensor_summation",id:"worked_examples/ex_einsum_old_tensor_summation",isDocsHomePage:!1,title:"ex_einsum_old_tensor_summation",description:"`py",source:"@site/docs/worked_examples/ex_einsum_old_tensor_summation.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_einsum_old_tensor_summation",permalink:"/csdl/docs/worked_examples/ex_einsum_old_tensor_summation",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_einsum_old_tensor_summation.mdx",tags:[],version:"current",frontMatter:{}},l=[],c={toc:l};function p(e){var n=e.components,t=(0,o.Z)(e,i);return(0,s.kt)("wrapper",(0,r.Z)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nfrom csdl import Model\nimport csdl\n\n\nclass ExampleTensorSummation(Model):\n    def define(self):\n        # Shape of Tensor\n        shape3 = (2, 4, 3)\n        c = np.arange(24).reshape(shape3)\n\n        # Declaring tensor\n        tens = self.declare_variable('c', val=c)\n\n        # Summation of all the entries of a tensor\n        self.register_output('einsum_summ2',\n                             csdl.einsum(\n                                 tens,\n                                 subscripts='ijk->',\n                             ))\n\n\nsim = Simulator(ExampleTensorSummation())\nsim.run()\n\nprint('c', sim['c'].shape)\nprint(sim['c'])\nprint('einsum_summ2', sim['einsum_summ2'].shape)\nprint(sim['einsum_summ2'])\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-c",metastring:"(2, 4, 3)","(2,":!0,"4,":!0,"3)":!0},"[[[ 0.  1.  2.]\n  [ 3.  4.  5.]\n  [ 6.  7.  8.]\n  [ 9. 10. 11.]]\n\n [[12. 13. 14.]\n  [15. 16. 17.]\n  [18. 19. 20.]\n  [21. 22. 23.]]]\neinsum_summ2 (1,)\n[276.]\n")))}p.isMDXComponent=!0}}]);