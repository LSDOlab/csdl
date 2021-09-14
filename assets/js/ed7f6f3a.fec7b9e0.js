"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[969],{3905:function(e,t,n){n.d(t,{Zo:function(){return c},kt:function(){return d}});var r=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function s(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var a=r.createContext({}),u=function(e){var t=r.useContext(a),n=t;return e&&(n="function"==typeof e?e(t):s(s({},t),e)),n},c=function(e){var t=u(e.components);return r.createElement(a.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,i=e.originalType,a=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),m=u(n),d=o,_=m["".concat(a,".").concat(d)]||m[d]||p[d]||i;return n?r.createElement(_,s(s({ref:t},c),{},{components:n})):r.createElement(_,s({ref:t},c))}));function d(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=n.length,s=new Array(i);s[0]=m;var l={};for(var a in t)hasOwnProperty.call(t,a)&&(l[a]=t[a]);l.originalType=e,l.mdxType="string"==typeof e?e:o,s[1]=l;for(var u=2;u<i;u++)s[u]=n[u];return r.createElement.apply(null,s)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},7200:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return l},contentTitle:function(){return a},metadata:function(){return u},toc:function(){return c},default:function(){return m}});var r=n(7462),o=n(3366),i=(n(7294),n(3905)),s=["components"],l={},a="Einstein Summation Numpy API",u={unversionedId:"std_lib_ref/Linear Algebra/einsum_old",id:"std_lib_ref/Linear Algebra/einsum_old",isDocsHomePage:!1,title:"Einstein Summation Numpy API",description:"This is evaluates the einstein summation on the operands.",source:"@site/docs/std_lib_ref/Linear Algebra/einsum_old.mdx",sourceDirName:"std_lib_ref/Linear Algebra",slug:"/std_lib_ref/Linear Algebra/einsum_old",permalink:"/csdl/docs/std_lib_ref/Linear Algebra/einsum_old",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/std_lib_ref/Linear Algebra/einsum_old.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Einstein Summation New API",permalink:"/csdl/docs/std_lib_ref/Linear Algebra/einsum_new"},next:{title:"Inner",permalink:"/csdl/docs/std_lib_ref/Linear Algebra/inner"}},c=[{value:"Inner Products",id:"inner-products",children:[]},{value:"Outer Products",id:"outer-products",children:[]},{value:"Reorder Operations",id:"reorder-operations",children:[]},{value:"Summation Operations",id:"summation-operations",children:[]},{value:"Special Operations",id:"special-operations",children:[]}],p={toc:c};function m(e){var t=e.components,n=(0,o.Z)(e,s);return(0,i.kt)("wrapper",(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"einstein-summation-numpy-api"},"Einstein Summation Numpy API"),(0,i.kt)("p",null,"This is evaluates the einstein summation on the operands.\nIt is analogous to numpy.einsum, and uses the same notation."),(0,i.kt)("p",null,"Examples for all the possible use cases are provided below."),(0,i.kt)("p",null,".. autofunction:: csdl.std.einsum.einsum"),(0,i.kt)("h2",{id:"inner-products"},"Inner Products"),(0,i.kt)("p",null,".. toctree::\n:maxdepth: 1\n:titlesonly:"),(0,i.kt)("p",null,"   ex_einsum_old_inner_vector_vector.rst\nex_einsum_old_inner_vector_vector_sparse.rst\nex_einsum_old_inner_tensor_vector.rst\nex_einsum_old_inner_tensor_vector_sparse.rst"),(0,i.kt)("h2",{id:"outer-products"},"Outer Products"),(0,i.kt)("p",null,".. toctree::\n:maxdepth: 1\n:titlesonly:"),(0,i.kt)("p",null,"   ex_einsum_old_outer_vector_vector.rst\nex_einsum_old_outer_vector_vector_sparse.rst\nex_einsum_old_outer_tensor_vector.rst\nex_einsum_old_outer_tensor_vector_sparse.rst"),(0,i.kt)("h2",{id:"reorder-operations"},"Reorder Operations"),(0,i.kt)("p",null,".. toctree::\n:maxdepth: 1\n:titlesonly:"),(0,i.kt)("p",null,"   ex_einsum_old_reorder_matrix.rst\nex_einsum_old_reorder_matrix_sparse.rst\nex_einsum_old_reorder_tensor.rst\nex_einsum_old_reorder_tensor_sparse.rst"),(0,i.kt)("h2",{id:"summation-operations"},"Summation Operations"),(0,i.kt)("p",null,".. toctree::\n:maxdepth: 1\n:titlesonly:"),(0,i.kt)("p",null,"   ex_einsum_old_vector_summation.rst\nex_einsum_old_vector_summation_sparse.rst\nex_einsum_old_tensor_summation.rst\nex_einsum_old_tensor_summation_sparse.rst"),(0,i.kt)("h2",{id:"special-operations"},"Special Operations"),(0,i.kt)("p",null,".. toctree::\n:maxdepth: 1\n:titlesonly:"),(0,i.kt)("p",null,"   ex_einsum_old_multiple_vector_summation.rst\nex_einsum_old_multiple_vector_summation_sparse.rst"))}m.isMDXComponent=!0}}]);