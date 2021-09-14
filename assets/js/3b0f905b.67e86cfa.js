"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[4553,5132],{3905:function(e,n,t){t.d(n,{Zo:function(){return l},kt:function(){return d}});var r=t(7294);function o(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function a(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?i(Object(t),!0).forEach((function(n){o(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function s(e,n){if(null==e)return{};var t,r,o=function(e,n){if(null==e)return{};var t,r,o={},i=Object.keys(e);for(r=0;r<i.length;r++)t=i[r],n.indexOf(t)>=0||(o[t]=e[t]);return o}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)t=i[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}var m=r.createContext({}),u=function(e){var n=r.useContext(m),t=n;return e&&(t="function"==typeof e?e(n):a(a({},n),e)),t},l=function(e){var n=u(e.components);return r.createElement(m.Provider,{value:n},e.children)},c={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},p=r.forwardRef((function(e,n){var t=e.components,o=e.mdxType,i=e.originalType,m=e.parentName,l=s(e,["components","mdxType","originalType","parentName"]),p=u(t),d=o,_=p["".concat(m,".").concat(d)]||p[d]||c[d]||i;return t?r.createElement(_,a(a({ref:n},l),{},{components:t})):r.createElement(_,a({ref:n},l))}));function d(e,n){var t=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var i=t.length,a=new Array(i);a[0]=p;var s={};for(var m in n)hasOwnProperty.call(n,m)&&(s[m]=n[m]);s.originalType=e,s.mdxType="string"==typeof e?e:o,a[1]=s;for(var u=2;u<i;u++)a[u]=t[u];return r.createElement.apply(null,a)}return r.createElement.apply(null,t)}p.displayName="MDXCreateElement"},4552:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return m},contentTitle:function(){return u},metadata:function(){return l},toc:function(){return c},default:function(){return d}});var r=t(7462),o=t(3366),i=(t(7294),t(3905)),a=t(2227),s=["components"],m={},u="Single Vector Summation using Einsum",l={unversionedId:"examples/Standard Library/einsum_old/ex_einsum_old_vector_summation",id:"examples/Standard Library/einsum_old/ex_einsum_old_vector_summation",isDocsHomePage:!1,title:"Single Vector Summation using Einsum",description:"This is an example of how to properly use the einsum function",source:"@site/docs/examples/Standard Library/einsum_old/ex_einsum_old_vector_summation.mdx",sourceDirName:"examples/Standard Library/einsum_old",slug:"/examples/Standard Library/einsum_old/ex_einsum_old_vector_summation",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_vector_summation",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/einsum_old/ex_einsum_old_vector_summation.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Single Tensor Summation using Einsum with Sparse Partials",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_tensor_summation_sparse"},next:{title:"Single Vector Summation using Einsum with Sparse Partials",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_vector_summation_sparse"}},c=[],p={toc:c};function d(e){var n=e.components,t=(0,o.Z)(e,s);return(0,i.kt)("wrapper",(0,r.Z)({},p,t,{components:n,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"single-vector-summation-using-einsum"},"Single Vector Summation using Einsum"),(0,i.kt)("p",null,"This is an example of how to properly use the einsum function\nto compute the summation of a single vector."),(0,i.kt)(a.default,{mdxType:"WorkedExample"}))}d.isMDXComponent=!0},2227:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return s},contentTitle:function(){return m},metadata:function(){return u},toc:function(){return l},default:function(){return p}});var r=t(7462),o=t(3366),i=(t(7294),t(3905)),a=["components"],s={},m=void 0,u={unversionedId:"worked_examples/ex_einsum_old_vector_summation",id:"worked_examples/ex_einsum_old_vector_summation",isDocsHomePage:!1,title:"ex_einsum_old_vector_summation",description:"`py",source:"@site/docs/worked_examples/ex_einsum_old_vector_summation.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_einsum_old_vector_summation",permalink:"/csdl/docs/worked_examples/ex_einsum_old_vector_summation",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_einsum_old_vector_summation.mdx",tags:[],version:"current",frontMatter:{}},l=[],c={toc:l};function p(e){var n=e.components,t=(0,o.Z)(e,a);return(0,i.kt)("wrapper",(0,r.Z)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nfrom csdl import Model\nimport csdl\n\n\nclass ExampleVectorSummation(Model):\n    def define(self):\n        a = np.arange(4)\n        vec = self.declare_variable('a', val=a)\n\n        # Summation of all the entries of a vector\n        self.register_output('einsum_summ1',\n                             csdl.einsum(\n                                 vec,\n                                 subscripts='i->',\n                             ))\n\n\nsim = Simulator(ExampleVectorSummation())\nsim.run()\n\nprint('a', sim['a'].shape)\nprint(sim['a'])\nprint('einsum_summ1', sim['einsum_summ1'].shape)\nprint(sim['einsum_summ1'])\n")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-a",metastring:"(4,)","(4,)":!0},"[0. 1. 2. 3.]\neinsum_summ1 (1,)\n[6.]\n")))}p.isMDXComponent=!0}}]);