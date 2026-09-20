const BLOCKED = 'script,iframe,object,embed,base,meta,link,style';

export function sanitizeHTML(html) {
  const template = document.createElement('template');
  template.innerHTML = String(html);
  template.content.querySelectorAll(BLOCKED).forEach(node => node.remove());
  template.content.querySelectorAll('*').forEach(node => {
    for (const attribute of [...node.attributes]) {
      const name = attribute.name.toLowerCase();
      if (name.startsWith('on') || name === 'srcdoc') node.removeAttribute(attribute.name);
      if ((name === 'href' || name === 'src') && attribute.value) {
        try {
          const url = new URL(attribute.value, location.origin);
          if (!['http:', 'https:'].includes(url.protocol)) node.removeAttribute(attribute.name);
        } catch {
          node.removeAttribute(attribute.name);
        }
      }
    }
    if (node.tagName === 'A' && node.getAttribute('target') === '_blank') {
      node.setAttribute('rel', 'noopener noreferrer');
    }
  });
  return template.innerHTML;
}

export function setSafeHTML(element, html) {
  element.innerHTML = sanitizeHTML(html);
}
