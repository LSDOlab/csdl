"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[1742,2167],{3905:function(e,n,r){r.d(n,{Zo:function(){return l},kt:function(){return d}});var t=r(7294);function o(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function s(e,n){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),r.push.apply(r,t)}return r}function i(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{};n%2?s(Object(r),!0).forEach((function(n){o(e,n,r[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):s(Object(r)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(r,n))}))}return e}function a(e,n){if(null==e)return{};var r,t,o=function(e,n){if(null==e)return{};var r,t,o={},s=Object.keys(e);for(t=0;t<s.length;t++)r=s[t],n.indexOf(r)>=0||(o[r]=e[r]);return o}(e,n);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(t=0;t<s.length;t++)r=s[t],n.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var u=t.createContext({}),c=function(e){var n=t.useContext(u),r=n;return e&&(r="function"==typeof e?e(n):i(i({},n),e)),r},l=function(e){var n=c(e.components);return t.createElement(u.Provider,{value:n},e.children)},m={inlineCode:"code",wrapper:function(e){var n=e.children;return t.createElement(t.Fragment,{},n)}},p=t.forwardRef((function(e,n){var r=e.components,o=e.mdxType,s=e.originalType,u=e.parentName,l=a(e,["components","mdxType","originalType","parentName"]),p=c(r),d=o,_=p["".concat(u,".").concat(d)]||p[d]||m[d]||s;return r?t.createElement(_,i(i({ref:n},l),{},{components:r})):t.createElement(_,i({ref:n},l))}));function d(e,n){var r=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var s=r.length,i=new Array(s);i[0]=p;var a={};for(var u in n)hasOwnProperty.call(n,u)&&(a[u]=n[u]);a.originalType=e,a.mdxType="string"==typeof e?e:o,i[1]=a;for(var c=2;c<s;c++)i[c]=r[c];return t.createElement.apply(null,i)}return t.createElement.apply(null,r)}p.displayName="MDXCreateElement"},7060:function(e,n,r){r.r(n),r.d(n,{frontMatter:function(){return u},contentTitle:function(){return c},metadata:function(){return l},toc:function(){return m},default:function(){return d}});var t=r(7462),o=r(3366),s=(r(7294),r(3905)),i=r(3230),a=["components"],u={},c="Tensor-Vector Outer Product using Einsum",l={unversionedId:"examples/Standard Library/einsum_old/ex_einsum_old_outer_tensor_vector",id:"examples/Standard Library/einsum_old/ex_einsum_old_outer_tensor_vector",isDocsHomePage:!1,title:"Tensor-Vector Outer Product using Einsum",description:"This is an example of how to properly use the einsum function",source:"@site/docs/examples/Standard Library/einsum_old/ex_einsum_old_outer_tensor_vector.mdx",sourceDirName:"examples/Standard Library/einsum_old",slug:"/examples/Standard Library/einsum_old/ex_einsum_old_outer_tensor_vector",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_outer_tensor_vector",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/einsum_old/ex_einsum_old_outer_tensor_vector.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Multiple Vector Summation using Einsum with Sparse Partials",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_multiple_vector_summation_sparse"},next:{title:"Tensor-Vector Outer Product using Einsum with Sparse Partials",permalink:"/csdl/docs/examples/Standard Library/einsum_old/ex_einsum_old_outer_tensor_vector_sparse"}},m=[],p={toc:m};function d(e){var n=e.components,r=(0,o.Z)(e,a);return(0,s.kt)("wrapper",(0,t.Z)({},p,r,{components:n,mdxType:"MDXLayout"}),(0,s.kt)("h1",{id:"tensor-vector-outer-product-using-einsum"},"Tensor-Vector Outer Product using Einsum"),(0,s.kt)("p",null,"This is an example of how to properly use the einsum function\nto compute a tensor-vector outer product."),(0,s.kt)(i.default,{mdxType:"WorkedExample"}))}d.isMDXComponent=!0},3230:function(e,n,r){r.r(n),r.d(n,{frontMatter:function(){return a},contentTitle:function(){return u},metadata:function(){return c},toc:function(){return l},default:function(){return p}});var t=r(7462),o=r(3366),s=(r(7294),r(3905)),i=["components"],a={},u=void 0,c={unversionedId:"worked_examples/ex_einsum_old_outer_tensor_vector",id:"worked_examples/ex_einsum_old_outer_tensor_vector",isDocsHomePage:!1,title:"ex_einsum_old_outer_tensor_vector",description:"`py",source:"@site/docs/worked_examples/ex_einsum_old_outer_tensor_vector.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_einsum_old_outer_tensor_vector",permalink:"/csdl/docs/worked_examples/ex_einsum_old_outer_tensor_vector",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_einsum_old_outer_tensor_vector.mdx",tags:[],version:"current",frontMatter:{}},l=[],m={toc:l};function p(e){var n=e.components,r=(0,o.Z)(e,i);return(0,s.kt)("wrapper",(0,t.Z)({},m,r,{components:n,mdxType:"MDXLayout"}),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nfrom csdl import Model\nimport csdl\n\n\nclass ExampleOuterTensorVector(Model):\n    def define(self):\n        a = np.arange(4)\n        vec = self.declare_variable('a', val=a)\n\n        # Shape of Tensor\n        shape3 = (2, 4, 3)\n        c = np.arange(24).reshape(shape3)\n\n        # Declaring tensor\n        tens = self.declare_variable('c', val=c)\n\n        # Outer Product of a tensor and a vector\n        self.register_output(\n            'einsum_outer2',\n            csdl.einsum(\n                tens,\n                vec,\n                subscripts='hij,k->hijk',\n            ))\n\n\nsim = Simulator(ExampleOuterTensorVector())\nsim.run()\n\nprint('a', sim['a'].shape)\nprint(sim['a'])\nprint('c', sim['c'].shape)\nprint(sim['c'])\nprint('einsum_outer2', sim['einsum_outer2'].shape)\nprint(sim['einsum_outer2'])\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-a",metastring:"(4,)","(4,)":!0},"[0. 1. 2. 3.]\nc (2, 4, 3)\n[[[ 0.  1.  2.]\n  [ 3.  4.  5.]\n  [ 6.  7.  8.]\n  [ 9. 10. 11.]]\n\n [[12. 13. 14.]\n  [15. 16. 17.]\n  [18. 19. 20.]\n  [21. 22. 23.]]]\neinsum_outer2 (2, 4, 3, 4)\n[[[[ 0.  0.  0.  0.]\n   [ 0.  1.  2.  3.]\n   [ 0.  2.  4.  6.]]\n\n  [[ 0.  3.  6.  9.]\n   [ 0.  4.  8. 12.]\n   [ 0.  5. 10. 15.]]\n\n  [[ 0.  6. 12. 18.]\n   [ 0.  7. 14. 21.]\n   [ 0.  8. 16. 24.]]\n\n  [[ 0.  9. 18. 27.]\n   [ 0. 10. 20. 30.]\n   [ 0. 11. 22. 33.]]]\n\n\n [[[ 0. 12. 24. 36.]\n   [ 0. 13. 26. 39.]\n   [ 0. 14. 28. 42.]]\n\n  [[ 0. 15. 30. 45.]\n   [ 0. 16. 32. 48.]\n   [ 0. 17. 34. 51.]]\n\n  [[ 0. 18. 36. 54.]\n   [ 0. 19. 38. 57.]\n   [ 0. 20. 40. 60.]]\n\n  [[ 0. 21. 42. 63.]\n   [ 0. 22. 44. 66.]\n   [ 0. 23. 46. 69.]]]]\n")))}p.isMDXComponent=!0}}]);