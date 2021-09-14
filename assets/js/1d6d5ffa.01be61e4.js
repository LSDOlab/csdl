"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[591],{3905:function(e,r,n){n.d(r,{Zo:function(){return m},kt:function(){return d}});var t=n(7294);function o(e,r,n){return r in e?Object.defineProperty(e,r,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[r]=n,e}function a(e,r){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);r&&(t=t.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),n.push.apply(n,t)}return n}function s(e){for(var r=1;r<arguments.length;r++){var n=null!=arguments[r]?arguments[r]:{};r%2?a(Object(n),!0).forEach((function(r){o(e,r,n[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))}))}return e}function i(e,r){if(null==e)return{};var n,t,o=function(e,r){if(null==e)return{};var n,t,o={},a=Object.keys(e);for(t=0;t<a.length;t++)n=a[t],r.indexOf(n)>=0||(o[n]=e[n]);return o}(e,r);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(t=0;t<a.length;t++)n=a[t],r.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var p=t.createContext({}),c=function(e){var r=t.useContext(p),n=r;return e&&(n="function"==typeof e?e(r):s(s({},r),e)),n},m=function(e){var r=c(e.components);return t.createElement(p.Provider,{value:r},e.children)},u={inlineCode:"code",wrapper:function(e){var r=e.children;return t.createElement(t.Fragment,{},r)}},l=t.forwardRef((function(e,r){var n=e.components,o=e.mdxType,a=e.originalType,p=e.parentName,m=i(e,["components","mdxType","originalType","parentName"]),l=c(n),d=o,_=l["".concat(p,".").concat(d)]||l[d]||u[d]||a;return n?t.createElement(_,s(s({ref:r},m),{},{components:n})):t.createElement(_,s({ref:r},m))}));function d(e,r){var n=arguments,o=r&&r.mdxType;if("string"==typeof e||o){var a=n.length,s=new Array(a);s[0]=l;var i={};for(var p in r)hasOwnProperty.call(r,p)&&(i[p]=r[p]);i.originalType=e,i.mdxType="string"==typeof e?e:o,s[1]=i;for(var c=2;c<a;c++)s[c]=n[c];return t.createElement.apply(null,s)}return t.createElement.apply(null,n)}l.displayName="MDXCreateElement"},1948:function(e,r,n){n.r(r),n.d(r,{frontMatter:function(){return i},contentTitle:function(){return p},metadata:function(){return c},toc:function(){return m},default:function(){return l}});var t=n(7462),o=n(3366),a=(n(7294),n(3905)),s=["components"],i={},p=void 0,c={unversionedId:"worked_examples/ex_einsum_new_reorder_matrix_sparse",id:"worked_examples/ex_einsum_new_reorder_matrix_sparse",isDocsHomePage:!1,title:"ex_einsum_new_reorder_matrix_sparse",description:"`py",source:"@site/docs/worked_examples/ex_einsum_new_reorder_matrix_sparse.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_einsum_new_reorder_matrix_sparse",permalink:"/csdl/docs/worked_examples/ex_einsum_new_reorder_matrix_sparse",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_einsum_new_reorder_matrix_sparse.mdx",tags:[],version:"current",frontMatter:{}},m=[],u={toc:m};function l(e){var r=e.components,n=(0,o.Z)(e,s);return(0,a.kt)("wrapper",(0,t.Z)({},u,n,{components:r,mdxType:"MDXLayout"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nfrom csdl import Model\nimport csdl\n\n\nclass ExampleReorderMatrixSparse(Model):\n    def define(self):\n\n        shape2 = (5, 4)\n        b = np.arange(20).reshape(shape2)\n        mat = self.declare_variable('b', val=b)\n\n        self.register_output(\n            'einsum_reorder1_sparse_derivs',\n            csdl.einsum_new_api(mat,\n                                operation=[(46, 99), (99, 46)],\n                                partial_format='sparse'))\n\n\nsim = Simulator(ExampleReorderMatrixSparse())\nsim.run()\n\nprint('b', sim['b'].shape)\nprint(sim['b'])\nprint('einsum_reorder1_sparse_derivs',\n      sim['einsum_reorder1_sparse_derivs'].shape)\nprint(sim['einsum_reorder1_sparse_derivs'])\n")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-b",metastring:"(5, 4)","(5,":!0,"4)":!0},"[[ 0.  1.  2.  3.]\n [ 4.  5.  6.  7.]\n [ 8.  9. 10. 11.]\n [12. 13. 14. 15.]\n [16. 17. 18. 19.]]\neinsum_reorder1_sparse_derivs (4, 5)\n[[ 0.  4.  8. 12. 16.]\n [ 1.  5.  9. 13. 17.]\n [ 2.  6. 10. 14. 18.]\n [ 3.  7. 11. 15. 19.]]\n")))}l.isMDXComponent=!0}}]);