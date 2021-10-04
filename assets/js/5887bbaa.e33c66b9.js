"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[7507],{3905:function(e,t,n){n.d(t,{Zo:function(){return d},kt:function(){return s}});var r=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var c=r.createContext({}),p=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},d=function(e){var t=p(e.components);return r.createElement(c.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,a=e.originalType,c=e.parentName,d=l(e,["components","mdxType","originalType","parentName"]),m=p(n),s=i,f=m["".concat(c,".").concat(s)]||m[s]||u[s]||a;return n?r.createElement(f,o(o({ref:t},d),{},{components:n})):r.createElement(f,o({ref:t},d))}));function s(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var a=n.length,o=new Array(a);o[0]=m;var l={};for(var c in t)hasOwnProperty.call(t,c)&&(l[c]=t[c]);l.originalType=e,l.mdxType="string"==typeof e?e:i,o[1]=l;for(var p=2;p<a;p++)o[p]=n[p];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},7032:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return l},contentTitle:function(){return c},metadata:function(){return p},toc:function(){return d},default:function(){return m}});var r=n(7462),i=n(3366),a=(n(7294),n(3905)),o=["components"],l={title:"Building a Compiler Back End",sidebar_position:5},c=void 0,p={unversionedId:"developer/contributing/backend",id:"developer/contributing/backend",isDocsHomePage:!1,title:"Building a Compiler Back End",description:"------------------------------------------------------------------------",source:"@site/docs/developer/contributing/backend.mdx",sourceDirName:"developer/contributing",slug:"/developer/contributing/backend",permalink:"/csdl/docs/developer/contributing/backend",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/developer/contributing/backend.mdx",tags:[],version:"current",sidebarPosition:5,frontMatter:{title:"Building a Compiler Back End",sidebar_position:5},sidebar:"docs",previous:{title:"Adding Standard Library Operations",permalink:"/csdl/docs/developer/contributing/std_lib"},next:{title:"Building and Updating Documentation",permalink:"/csdl/docs/developer/contributing/docs"}},d=[],u={toc:d};function m(e){var t=e.components,n=(0,i.Z)(e,o);return(0,a.kt)("wrapper",(0,r.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("hr",null),(0,a.kt)("p",null,"The CSDL compiler back end ",(0,a.kt)("em",{parentName:"p"},"must")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"Provide a class named ",(0,a.kt)("inlineCode",{parentName:"li"},"Simulator")," that conforms to the ",(0,a.kt)("inlineCode",{parentName:"li"},"SimulatorBase"),"\nAPI (see ",(0,a.kt)("a",{parentName:"li",href:"../api"},"Developer API"),").",(0,a.kt)("ul",{parentName:"li"},(0,a.kt)("li",{parentName:"ul"},"The ",(0,a.kt)("inlineCode",{parentName:"li"},"Simulator")," class ",(0,a.kt)("em",{parentName:"li"},"should")," inherit from ",(0,a.kt)("inlineCode",{parentName:"li"},"SimulatorBase")),(0,a.kt)("li",{parentName:"ul"},"The ",(0,a.kt)("inlineCode",{parentName:"li"},"Simulator")," class constructor ",(0,a.kt)("em",{parentName:"li"},"must")," have a ",(0,a.kt)("inlineCode",{parentName:"li"},"Model")," instance as\nan unnamed argument and call the ",(0,a.kt)("inlineCode",{parentName:"li"},"Model.define")," method.\nThe user does not need to ever call ",(0,a.kt)("inlineCode",{parentName:"li"},"Model.define"),"."))),(0,a.kt)("li",{parentName:"ul"},"Implement the entire Standard Library, and the partial derivatives for\neach standard operation."),(0,a.kt)("li",{parentName:"ul"},"Implement the MAUD architecture."),(0,a.kt)("li",{parentName:"ul"},"Not store a ",(0,a.kt)("inlineCode",{parentName:"li"},"Model")," object; i.e. deletion of the ",(0,a.kt)("inlineCode",{parentName:"li"},"Model")," object\n",(0,a.kt)("em",{parentName:"li"},"must be allowed")," after a ",(0,a.kt)("inlineCode",{parentName:"li"},"Simulator")," object is constructed, imposing\nno additional memory overhead when ",(0,a.kt)("inlineCode",{parentName:"li"},"Simulator.run")," is called.")),(0,a.kt)("p",null,"The CSDL compiler back end ",(0,a.kt)("em",{parentName:"p"},"should"),"\nThe ",(0,a.kt)("inlineCode",{parentName:"p"},"Simulator")," class can be implemented in any language, as long as\nthere is a Python interface available.\nFor example, if ",(0,a.kt)("inlineCode",{parentName:"p"},"Simulator")," is implemented in C++, the ",(0,a.kt)("inlineCode",{parentName:"p"},"pybind11"),"\nlibrary can expose the C++ class to Python."))}m.isMDXComponent=!0}}]);