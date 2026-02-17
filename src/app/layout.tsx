import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Kineforge',
  description: 'Node-based live media forge for choreography, AI vision, and GPU interaction prototyping.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
