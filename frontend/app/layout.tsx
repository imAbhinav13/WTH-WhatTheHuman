import type { Metadata, Viewport } from "next";
import { EB_Garamond } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

const ebGaramond = EB_Garamond({
  subsets: ["latin"],
  variable: "--font-eb-garamond",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "WTH - What The Human",
    template: "%s · WTH - What The Human",
  },
  description:
    "A citation-grounded reading experience comparing modern science, Advaita Vedanta, and Samkhya on questions of consciousness, self, and experienced reality.",
  openGraph: {
    title: "WTH — What The Human",
    description:
      "One question, three independently grounded perspectives, and an explicit comparison of where they overlap, differ, or cannot be compared.",
    type: "website",
  },
  twitter: {
    card: "summary",
    title: "WTH — What The Human",
    description:
      "A citation-grounded comparative reading experience for questions about consciousness, self, and reality.",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#EFE3D1",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={ebGaramond.variable}>
      <body>
        <div className="min-h-screen">
          <Header />
          <main className="mx-auto w-full max-w-reading px-5 pb-16 pt-8 sm:px-8 sm:pt-12">
            {children}
          </main>
          <Footer />
        </div>
      </body>
    </html>
  );
}
